"""RAFE deployment entry points; simulator imports are deferred until evaluation."""


def get_model(usr_args):
    from .deploy_policy import get_model as build_model
    return build_model(usr_args)


def eval(TASK_ENV, model, observation):
    from .deploy_policy import eval as evaluate
    return evaluate(TASK_ENV, model, observation)


def reset_model(model):
    from .deploy_policy import reset_model as reset
    return reset(model)
