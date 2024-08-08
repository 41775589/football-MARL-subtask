
import argparse
import csv
import logging
import os
import sys
import time

import ray
from utils import saving
import tf_models
import yaml
from env_wrapper import RllibGFootball
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.logger import NoopLogger, pretty_print

ray.init(log_to_driver=False)

logging.basicConfig(stream=sys.stdout, format="%(asctime)s %(message)s")
logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)


def process_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir", type=str, help="Path to the directory for this run."
    )
    parser.add_argument('--num-agents', type=int, default=5)
    parser.add_argument('--num-policies', type=int, default=5)
    parser.add_argument('--num-iters', type=int, default=100000)

    args = parser.parse_args()
    run_directory = args.run_dir

    config_path = os.path.join(args.run_dir, "config.yaml")
    assert os.path.isdir(args.run_dir)
    assert os.path.isfile(config_path)

    with open(config_path, "r") as f:
        run_configuration = yaml.safe_load(f)

    num_agents = args.num_agents
    num_policies = args.num_policies
    num_iters = args.num_iters


    return run_directory, run_configuration, num_agents, num_policies, num_iters


def build_trainer(run_configuration,num_agents, num_policies):
    """Finalize the trainer config by combining the sub-configs."""
    trainer_config = run_configuration.get("trainer")

    # === Env ===
    env_config = {
        "env_config_dict": run_configuration.get("env"),
        "num_envs_per_worker": trainer_config.get("num_envs_per_worker"),
    }


    single_env = RllibGFootball(env_config)
    obs_space = single_env.observation_space
    act_space = single_env.action_space

    def gen_policy(_):
        return (None, obs_space, act_space, {})

    policies = {
        'policy_{}'.format(i): gen_policy(i) for i in range(num_policies)
    }
    policy_ids = list(policies.keys())



    # === Finalize and create ===
    trainer_config.update(
        {
            "env_config": env_config,
            "multiagent": {
                "policies": policies,
                'policy_mapping_fn': lambda agent_id: policy_ids[int(agent_id[6:])]
            },
            # "metrics_smoothing_episodes": trainer_config.get("num_workers")
            # * trainer_config.get("num_envs_per_worker"),
        }
    )

    def logger_creator(config):
        return NoopLogger({}, "/tmp")

    ppo_trainer = PPOTrainer(
        env=RllibGFootball, config=trainer_config,
        # logger_creator=logger_creator
    )

    return ppo_trainer


def set_up_dirs_and_maybe_restore(run_directory, run_configuration, trainer_obj, num_policies):
    # === Set up Logging & Saving, or Restore ===
    # All model parameters are always specified in the settings YAML.
    # We do NOT overwrite / reload settings from the previous checkpoint dir.
    # 1. For new runs, the only object that will be loaded from the checkpoint dir
    #    are model weights.
    # 2. For crashed and restarted runs, load_snapshot will reload the full state of
    #    the Trainer(s), including metadata, optimizer, and models.
    (
        ckpt_directory,
        restore_from_crashed_run,
    ) = saving.fill_out_run_dir(run_directory)

    # If this is a starting from a crashed run, restore the last trainer snapshot
    if restore_from_crashed_run:
        logger.info(
            "ckpt_dir already exists! Planning to restore using latest snapshot from "
            "earlier (crashed) run with the same ckpt_dir %s",
            ckpt_directory,
        )


        # === Trainer-specific counters ===
        training_step_last_ckpt = 0
        iters_last_ckpt = 0

    else:
        logger.info("Not restoring trainer...")
        # === Trainer-specific counters ===
        training_step_last_ckpt = 0
        iters_last_ckpt = 0


        for i in range(num_policies):
            starting_weights_path = run_configuration["general"].get(
                "restore_tf_weights_{}".format(i), ""
            )
            if starting_weights_path:
                logger.info("Restoring policy_{} TF weights...".format(i))
                saving.load_tf_model_weights(trainer_obj, starting_weights_path)
            else:
                logger.info("Starting with fresh policy_{} TF weights.".format(i))


    return (
        ckpt_directory,
        restore_from_crashed_run,
        training_step_last_ckpt,
        iters_last_ckpt,
    )



def maybe_save(trainer_obj, result_dict, ckpt_freq, ckpt_directory, trainer_step_last_ckpt, num_policies):
    global_step = result_dict["timesteps_total"]

    # Check if saving this iteration
    if (
        result_dict["training_iteration"] > 0
    ):  # Don't save if midway through an iter.

        if ckpt_freq > 0:
            if global_step - trainer_step_last_ckpt >= ckpt_freq:
                # saving.save_snapshot(trainer_obj, ckpt_directory, suffix="")
                for i in range(num_policies):
                    saving.save_tf_model_weights(
                        trainer_obj, ckpt_directory, global_step, suffix='policy_{}'.format(i)
                    )

                trainer_step_last_ckpt = int(global_step)

                logger.info("Checkpoint saved @ step %d", global_step)

    return trainer_step_last_ckpt


if __name__ == "__main__":

    with open("train_result.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['time_step',
                        'iter',
                         'episodes',
                         'episode_reward_max',
                         'episode_reward_min',
                         'episode_reward_mean',

                         'policy_loss_0',
                         'total_loss_0',
                         'policy_loss_1',
                         'total_loss_1',
                         'policy_loss_2',
                         'total_loss_2',
                         'policy_loss_3',
                         'total_loss_3',
                         'policy_loss_4',
                         'total_loss_4',

                         'policy_reward_max_0',
                         'policy_reward_max_1',
                         'policy_reward_max_2',
                         'policy_reward_max_3',
                         'policy_reward_max_4',

                         'policy_reward_min_0',
                         'policy_reward_min_1',
                         'policy_reward_min_2',
                         'policy_reward_min_3',
                         'policy_reward_min_4',

                         'policy_reward_mean_0',
                         'policy_reward_mean_1',
                         'policy_reward_mean_2',
                         'policy_reward_mean_3',
                         'policy_reward_mean_4',
                         ])


    # ===================
    # === Start setup ===
    # ===================

    # Process the args
    run_dir, run_config, num_agents, num_policies, num_iters = process_args()

    # Create a trainer object
    trainer = build_trainer(run_config, num_agents, num_policies)

    # Set up directories for logging and saving. Restore if this has already been
    # done (indicating that we're restarting a crashed run). Or, if appropriate,
    # load in starting model weights for the agent and/or planner.
    (
        ckpt_dir,
        restore_from_crashed_run,
        step_last_ckpt,
        curr_iter,
    ) = set_up_dirs_and_maybe_restore(run_dir, run_config, trainer, num_policies)

    # ======================
    # === Start training ===
    # ======================

    ckpt_frequency = run_config["general"].get("ckpt_frequency_steps", 0)
    global_step = int(step_last_ckpt)

    while curr_iter < run_config["general"]["iters"]:

        # Training
        result = trainer.train()


        # === Counters++ ===
        global_step = result["timesteps_total"]
        curr_iter = result["training_iteration"]

        logger.info(
            "Iter %d: total steps %d -> %d/%d iterations done",
            curr_iter,
            global_step,
            curr_iter,
            run_config["general"]["iters"],
        )



        logger.info(pretty_print(result))

        if result.get('episode_reward_max') != {}:
            with open("train_result.csv", "a+", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                  global_step,
                  curr_iter,
                    result.get('episodes_total'),
                  result.get('episode_reward_max'),
                  result.get('episode_reward_min'),
                  result.get('episode_reward_mean'),

                  result.get('info').get('learner').get('policy_0').get('policy_loss'),
                  result.get('info').get('learner').get('policy_0').get('total_loss'),
                  result.get('info').get('learner').get('policy_1').get('policy_loss'),
                  result.get('info').get('learner').get('policy_1').get('total_loss'),
                    result.get('info').get('learner').get('policy_2').get('policy_loss'),
                    result.get('info').get('learner').get('policy_2').get('total_loss'),
                    result.get('info').get('learner').get('policy_3').get('policy_loss'),
                    result.get('info').get('learner').get('policy_3').get('total_loss'),
                    result.get('info').get('learner').get('policy_4').get('policy_loss'),
                    result.get('info').get('learner').get('policy_4').get('total_loss'),

                  result.get('policy_reward_max').get('policy_0'),
                  result.get('policy_reward_max').get('policy_1'),
                result.get('policy_reward_max').get('policy_2'),
                  result.get('policy_reward_max').get('policy_3'),
                result.get('policy_reward_max').get('policy_4'),

                  result.get('policy_reward_min').get('policy_0'),
                  result.get('policy_reward_min').get('policy_1'),
                    result.get('policy_reward_min').get('policy_2'),
                    result.get('policy_reward_min').get('policy_3'),
                    result.get('policy_reward_min').get('policy_4'),

                  result.get('policy_reward_mean').get('policy_0'),
                  result.get('policy_reward_mean').get('policy_1'),
                result.get('policy_reward_mean').get('policy_2'),
                result.get('policy_reward_mean').get('policy_3'),
                result.get('policy_reward_mean').get('policy_4')
                ])

        # === Saving ===
        step_last_ckpt = maybe_save(
            trainer, result, ckpt_frequency, ckpt_dir, step_last_ckpt, num_policies
        )

    # Finish up
    logger.info("Completing! Saving final model weights...\n\n")
    # saving.save_snapshot(trainer, ckpt_dir)


    for i in range(num_policies):
        saving.save_tf_model_weights(trainer, ckpt_dir, global_step, suffix='policy_{}'.format(i))

    logger.info("Final snapshot saved! All done.")

    ray.shutdown()  # shutdown Ray after use
