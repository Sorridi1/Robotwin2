"""Synchronous PPO collection with independently running simulation workers."""

import importlib
import multiprocessing as mp
from multiprocessing import shared_memory
import os
import time
import traceback

import numpy as np


def _instantiate(config):
    config = dict(config)
    module, name = config.pop("_target_").rsplit(".", 1)
    return getattr(importlib.import_module(module), name)(**config)


def _step_and_reset(env, action):
    started = time.perf_counter()
    obs, reward, terminated, truncated, info = env.step(action)
    info = dict(info)
    info["perf/env_step_sec"] = time.perf_counter() - started
    info["perf/reset_sec"] = 0.0
    if terminated or truncated:
        started = time.perf_counter()
        obs = env.reset()
        info["perf/reset_sec"] = time.perf_counter() - started
        info.update(getattr(env, "last_reset_info", {}))
    return obs, reward, terminated, truncated, info


class SerialEnvPool:
    def __init__(self, configs):
        self.envs = []
        try:
            for config in configs:
                self.envs.append(_instantiate(config))
        except BaseException:
            self.close()
            raise

    def __len__(self):
        return len(self.envs)

    def reset(self):
        observations = [env.reset() for env in self.envs]
        self.last_reset_infos = [dict(getattr(env, "last_reset_info", {})) for env in self.envs]
        return observations

    def step(self, actions):
        if len(actions) != len(self):
            raise ValueError("Action batch size does not match environment count")
        return [_step_and_reset(env, action) for env, action in zip(self.envs, actions)]

    def get_next_seeds(self):
        return [int(env.next_seed) for env in self.envs]

    def close(self, clear_cache=True):
        for env in self.envs:
            env.close(clear_cache=clear_cache)
        self.envs = []


def _worker(connection, config, index, memory_specs, threads):
    env = None
    handles = []
    try:
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[variable] = str(threads)
        # The real simulator imports torch; fake CPU-only environments need not.
        env = _instantiate(config)
        import sys
        if "torch" in sys.modules:
            sys.modules["torch"].set_num_threads(threads)
        arrays = {}
        for key, spec in memory_specs.items():
            handle = shared_memory.SharedMemory(name=spec["name"])
            handles.append(handle)
            arrays[key] = np.ndarray(spec["shape"], dtype=np.float32, buffer=handle.buf)

        def publish(obs):
            if not arrays:
                return obs
            for key, array in arrays.items():
                value = np.asarray(obs[key], dtype=np.float32)
                if value.shape != array[index].shape:
                    raise ValueError(f"{key}: expected {array[index].shape}, got {value.shape}")
                array[index] = value
            return None

        connection.send((True, None))
        while True:
            command, argument = connection.recv()
            if command == "reset":
                result = publish(env.reset()), dict(getattr(env, "last_reset_info", {}))
            elif command == "step":
                obs, reward, terminated, truncated, info = _step_and_reset(env, argument)
                result = publish(obs), reward, terminated, truncated, info
            elif command == "next_seed":
                result = int(env.next_seed)
            elif command == "close":
                env.close(clear_cache=bool(argument))
                env = None
                connection.send((True, None))
                break
            else:
                raise ValueError(f"Unknown environment worker command: {command}")
            connection.send((True, result))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        try:
            connection.send((False, traceback.format_exc()))
        except (EOFError, BrokenPipeError):
            pass
    finally:
        if env is not None:
            try:
                env.close(clear_cache=True)
            except Exception:
                pass
        for handle in handles:
            handle.close()
        connection.close()


class SubprocessEnvPool:
    def __init__(self, configs, observation_shapes=None, worker_threads=1, timeout_sec=1800):
        if not configs or worker_threads < 1 or timeout_sec <= 0:
            raise ValueError("Environment count, worker threads and timeout must be positive")
        self.count = len(configs)
        self.timeout_sec = float(timeout_sec)
        self.connections = []
        self.processes = []
        self.memory = []
        self.arrays = {}
        self.closed = False
        context = mp.get_context("spawn")
        specs = {}
        try:
            for key, shape in (observation_shapes or {}).items():
                shape = (self.count, *shape)
                memory = shared_memory.SharedMemory(create=True, size=int(np.prod(shape)) * 4)
                self.memory.append(memory)
                self.arrays[key] = np.ndarray(shape, dtype=np.float32, buffer=memory.buf)
                specs[key] = {"name": memory.name, "shape": shape}
            for index, config in enumerate(configs):
                parent, child = context.Pipe()
                process = context.Process(
                    target=_worker,
                    args=(child, config, index, specs, int(worker_threads)),
                    name=f"robotwin-env-{index}",
                    daemon=False,  # Some embodiments start their own planner processes.
                )
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            for index in range(self.count):
                self._receive(index)
        except BaseException:
            self.close(force=True)
            raise

    def __len__(self):
        return self.count

    def _receive(self, index):
        connection = self.connections[index]
        if not connection.poll(self.timeout_sec):
            raise TimeoutError(f"RoboTwin worker {index} timed out after {self.timeout_sec}s")
        try:
            success, result = connection.recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError(f"RoboTwin worker {index} exited unexpectedly") from exc
        if not success:
            raise RuntimeError(f"RoboTwin worker {index} failed:\n{result}")
        return result

    def _request_all(self, command, arguments):
        if self.closed:
            raise RuntimeError("Environment pool is closed")
        try:
            for connection, argument in zip(self.connections, arguments):
                connection.send((command, argument))
            return [self._receive(index) for index in range(self.count)]
        except BaseException:
            self.close(force=True)
            raise

    def _observation(self, index, value):
        if self.arrays:
            # The next worker step overwrites shared storage; buffers own their observations.
            return {key: array[index].copy() for key, array in self.arrays.items()}
        return value

    def reset(self):
        values = self._request_all("reset", [None] * self.count)
        self.last_reset_infos = [value[1] for value in values]
        return [self._observation(index, value[0]) for index, value in enumerate(values)]

    def step(self, actions):
        if len(actions) != self.count:
            raise ValueError("Action batch size does not match environment count")
        values = self._request_all("step", actions)
        return [
            (self._observation(index, value[0]), *value[1:])
            for index, value in enumerate(values)
        ]

    def get_next_seeds(self):
        return self._request_all("next_seed", [None] * self.count)

    def close(self, clear_cache=True, force=False):
        if self.closed:
            return
        self.closed = True
        if not force:
            for connection in self.connections:
                try:
                    connection.send(("close", clear_cache))
                except (BrokenPipeError, EOFError, OSError):
                    pass
        deadline = time.monotonic() + (0.0 if force else 10.0)
        for process in self.processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=2.0)
        for connection in self.connections:
            connection.close()
        self.arrays.clear()
        for memory in self.memory:
            memory.close()
            memory.unlink()
        self.memory.clear()
