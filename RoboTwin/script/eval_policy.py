import sys
import os
import subprocess

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

from generate_episode_instructions import *
from script.eval_scene_selection import (
    EvalSceneSelector,
    build_scene_cache,
    check_selection_mode,
    evaluation_result_dir,
    load_rl_scene_validation,
    load_scene_manifest,
    scene_validator_from_config,
)
from script.eval_seed_bank import DEFAULT_SHARED_SEED_FILE, SharedEvalSeedBank
from script.eval_baseline import DEFAULT_BASELINE_CHECKPOINT, FixedBaselineProbe
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def set_episode_instruction(task, task_name, episode_info, instruction_type, count):
    results = generate_episode_descriptions(task_name, [episode_info["info"]], count)
    if not results or not results[0].get(instruction_type):
        raise RuntimeError(
            "No evaluation instructions match the scene metadata. Implement "
            "get_scene_metadata() for this task instead of running the expert for metadata."
        )
    task.set_instruction(instruction=np.random.choice(results[0][instruction_type]))


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    save_dir = None
    video_save_dir = None
    video_size = None

    get_model = eval_function_decorator(policy_name, "get_model")
    

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])


    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = int(usr_args.get("eval_num_episodes", 100))
    max_seed_tries = int(usr_args.get("eval_max_seed_tries", 10000))
    scene_selection = check_selection_mode(usr_args.get("eval_scene_selection", "expert"))
    if test_num < 1 or max_seed_tries < 1:
        raise ValueError("eval_num_episodes and eval_max_seed_tries must be positive")
    if max_seed_tries < test_num:
        raise ValueError("eval_max_seed_tries must be at least eval_num_episodes")
    topk = 1

    model = get_model(usr_args)
    baseline = None
    if scene_selection == "baseline_fail":
        baseline = FixedBaselineProbe(
            usr_args.get("eval_baseline_checkpoint", DEFAULT_BASELINE_CHECKPOINT),
            lambda task, info: set_episode_instruction(task, task_name, info, instruction_type, test_num),
            task_name, device=usr_args.get("device", "cuda:0"),
        )
        print(f"Fixed baseline for scene selection: {baseline.checkpoint}", flush=True)
    manifest_path = usr_args.get("eval_scene_manifest")
    seed_bank = None
    if manifest_path in (None, "", "null", "None"):
        seed_bank = SharedEvalSeedBank(
            usr_args.get("eval_scene_seed_file", DEFAULT_SHARED_SEED_FILE), args, scene_selection, st_seed,
            baseline_identity=baseline.identity if baseline is not None else None,
        )
        print(f"Shared evaluation seed file: {seed_bank.path}; group={seed_bank.key}", flush=True)
    frozen, shared_validation = seed_bank.validation_config() if seed_bank is not None else (False, None)
    validator, validation_source, cache = None, None, None
    if scene_selection != "expert":
        explicit_validation = usr_args.get("eval_scene_validation_config")
        if frozen and explicit_validation in (None, "", "null", "None"):
            validator = scene_validator_from_config(shared_validation)
            validation_source = f"{seed_bank.path}#{seed_bank.key}"
        else:
            validator, validation_source = load_rl_scene_validation(
                getattr(model, "run_dir", None), explicit_validation,
            )
        cache = build_scene_cache(
            args, scene_selection, validator,
            usr_args.get("eval_scene_cache_dir", "/media/Elements1/ljj/ManiFlow/cache/eval_scenes"),
            usr_args.get("eval_scene_cache_mode", "readwrite"),
            baseline.identity if baseline is not None else None,
        )
        print(f"RL scene validation rules: {validation_source}", flush=True)
    if hasattr(model, "run_dir"):
        save_dir = Path(model.run_dir)
        epoch = model.epoch
        # add eval_results and epoch to save_dir
        save_dir = save_dir / "eval_results" / f"epoch_{epoch}"
    else:
        save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
    save_dir = evaluation_result_dir(save_dir, scene_selection)
    print(f"Evaluation scene selection: {scene_selection}; results: {save_dir}", flush=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    selector = EvalSceneSelector(
        scene_selection, args, UnStableError, validator, cache,
        baseline_probe=baseline, baseline_identity=baseline.identity if baseline is not None else None,
    )
    manifest_seeds = load_scene_manifest(
        manifest_path, selector, task_name, task_config, test_num,
    )

    st_seed, suc_num = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=test_num,
                                   video_size=video_size,
                                   instruction_type=instruction_type,
                                   selector=selector,
                                   report_path=save_dir / "_scene_selection.json",
                                   validation_source=validation_source,
                                   manifest_seeds=manifest_seeds,
                                   seed_bank=seed_bank,
                                   max_seed_tries=max_seed_tries)
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        file.write(f"Scene Selection: {scene_selection}\n\n")
        file.write(f"Episodes: {test_num}; Candidate Seeds: {selector.stats['attempts']}\n\n")
        if seed_bank is not None:
            file.write(f"Shared Seed File: {seed_bank.path}; Group: {seed_bank.key}\n\n")
        if baseline is not None:
            file.write(f"Fixed Baseline: {baseline.checkpoint}\n\n")
        # file.write(str(task_reward) + '\n')
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    print(f"Data has been saved to {file_path}")
    # return task_reward


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None,
                selector=None,
                report_path=None,
                validation_source=None,
                manifest_seeds=None,
                seed_bank=None,
                max_seed_tries=10000):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    selector = selector or EvalSceneSelector("expert", args, UnStableError)
    if seed_bank is not None and manifest_seeds is not None:
        raise ValueError("Use either the shared seed bank or an explicit manifest, not both")
    if seed_bank is not None:
        selector.shared_seed_file = str(seed_bank.path)
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed
    report_start_seed = manifest_seeds[0] if manifest_seeds is not None else st_seed
    if seed_bank is not None:
        report_start_seed = seed_bank.start_seed

    args["eval_mode"] = True

    status = "aborted"
    try:
        selector.save_report(report_path, task_name, args["task_config"], test_num,
                             report_start_seed, validation_source)
        while TASK_ENV.test_num < test_num:
            if selector.stats["attempts"] >= max_seed_tries:
                raise RuntimeError(
                    f"Selected only {TASK_ENV.test_num}/{test_num} evaluation episodes after "
                    f"{max_seed_tries} candidates in mode={selector.mode}; "
                    f"rejections={selector.stats['rejections']}. No partial success rate is reported."
                )
            if seed_bank is not None:
                selected = seed_bank.prepare(
                    TASK_ENV, selector, TASK_ENV.test_num, now_id,
                    max_candidates=max_seed_tries - selector.stats["attempts"],
                )
                now_seed = selected.record["seed"]
            else:
                if manifest_seeds is not None:
                    now_seed = manifest_seeds[TASK_ENV.test_num]
                selected = selector.prepare(TASK_ENV, now_seed, now_id)
            if selected is None:
                if manifest_seeds is not None:
                    raise RuntimeError(f"Manifest seed {now_seed} was rejected; refusing to replace the fixed cohort")
                now_seed += 1
                continue

            selector.save_report(report_path, task_name, args["task_config"], test_num,
                                 report_start_seed, validation_source)
            set_episode_instruction(TASK_ENV, task_name, selected.episode_info, instruction_type, test_num)

            if TASK_ENV.eval_video_path is not None:
                ffmpeg = subprocess.Popen(
                    [
                        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                        "-pixel_format", "rgb24", "-video_size", video_size,
                        "-framerate", "10", "-i", "-", "-pix_fmt", "yuv420p",
                        "-vcodec", "libx264", "-crf", "23",
                        f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                    ],
                    stdin=subprocess.PIPE,
                )
                TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

            succ = False
            try:
                reset_func(model)
                while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
                    observation = TASK_ENV.get_obs()
                    eval_func(TASK_ENV, model, observation)
                    if TASK_ENV.eval_success:
                        succ = True
                        break
            finally:
                if TASK_ENV.eval_video_path is not None:
                    TASK_ENV._del_eval_video_ffmpeg()

            selected.record["success"] = bool(succ)
            if succ:
                TASK_ENV.suc += 1
                print("\033[92mSuccess!\033[0m")
            else:
                print("\033[91mFail!\033[0m")

            now_id += 1
            if TASK_ENV.render_freq:
                TASK_ENV.viewer.close()
            selector.close(TASK_ENV)
            TASK_ENV.test_num += 1

            print(
                f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | "
                f"\033[92m{args['task_config']}\033[0m | scenes={selector.mode}\n"
                f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => "
                f"\033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, "
                f"current seed: \033[90m{now_seed}\033[0m\n", flush=True,
            )
            selector.save_report(report_path, task_name, args["task_config"], test_num,
                                 report_start_seed, validation_source)
            now_seed += 1
        status = "complete"
    finally:
        try:
            selector.close(TASK_ENV)
        finally:
            selector.save_report(report_path, task_name, args["task_config"], test_num,
                                 report_start_seed, validation_source, status=status)

    return now_seed, TASK_ENV.suc


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
