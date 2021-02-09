# -*- coding: utf-8 -*-


import argparse
import copy
import os
import random
import sys

import numpy as np
from tqdm import tqdm

from build_tree import (build_center_single, build_distribute_four,
                        build_distribute_nine,
                        build_in_center_single_out_center_single,
                        build_in_distribute_four_out_center_single,
                        build_left_center_single_right_center_single,
                        build_up_center_single_down_center_single)
from const import IMAGE_SIZE, RULE_ATTR
from rendering import (generate_matrix, generate_matrix_answer, imsave, imshow,
                       render_panel)
from Rule import Rule_Wrapper
from sampling import sample_attr, sample_attr_avail, sample_rules
from serialize import dom_problem, serialize_aot, serialize_rules
from solver import solve

import time
if os.name == 'nt':
    from eventlet.timeout import Timeout as timeout
else:
    import signal
    from contextlib import contextmanager


    def raise_timeout(signum, frame):
        raise TimeoutError


    @contextmanager
    def timeout(time):
        # Register a function to raise a TimeoutError on the signal.
        signal.signal(signal.SIGALRM, raise_timeout)
        # Schedule the signal to be sent after ``time``.
        signal.alarm(time)

        try:
            yield
        except Exception as e:
            raise e
            signal.signal(signal.SIGALRM, signal.SIG_IGN)


def merge_component(dst_aot, src_aot, component_idx):
    src_component = src_aot.children[0].children[component_idx]
    dst_aot.children[0].children[component_idx] = src_component


def separate(args, all_configs):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for key in all_configs.keys():
        if not os.path.exists(os.path.join(args.save_dir, key)):
            os.mkdir(os.path.join(args.save_dir, key))
            
    random.seed(args.seed)
    np.random.seed(args.seed)

    for key in all_configs.keys():
        acc = 0
        for k in tqdm(range(args.num_samples), key):
            count_num = k % 10
            if count_num < (10 - args.val - args.test):
                set_name = "train"
            elif count_num < (10 - args.test):
                set_name = "val"
            else:
                set_name = "test"

            root = all_configs[key]
            while True:
                rule_groups = sample_rules()
                new_root = root.prune(rule_groups)
                if new_root is not None:
                    break
            
            start_node = new_root.sample()

            row_1_1 = copy.deepcopy(start_node)
            for l in range(len(rule_groups)):
                rule_group = rule_groups[l]
                rule_num_pos = rule_group[0]
                row_1_2 = rule_num_pos.apply_rule(row_1_1)
                row_1_3 = rule_num_pos.apply_rule(row_1_2)
                for i in range(1, len(rule_group)):
                    rule = rule_group[i]
                    row_1_2 = rule.apply_rule(row_1_1, row_1_2)
                for i in range(1, len(rule_group)):
                    rule = rule_group[i]
                    row_1_3 = rule.apply_rule(row_1_2, row_1_3)
                if l == 0:
                    to_merge = [row_1_1, row_1_2, row_1_3]
                else:
                    merge_component(to_merge[1], row_1_2, l)
                    merge_component(to_merge[2], row_1_3, l)
            row_1_1, row_1_2, row_1_3 = to_merge

            row_2_1 = copy.deepcopy(start_node)
            row_2_1.resample(True)
            for l in range(len(rule_groups)):
                rule_group = rule_groups[l]
                rule_num_pos = rule_group[0]
                row_2_2 = rule_num_pos.apply_rule(row_2_1)
                row_2_3 = rule_num_pos.apply_rule(row_2_2)
                for i in range(1, len(rule_group)):
                    rule = rule_group[i]
                    row_2_2 = rule.apply_rule(row_2_1, row_2_2)
                for i in range(1, len(rule_group)):
                    rule = rule_group[i]
                    row_2_3 = rule.apply_rule(row_2_2, row_2_3)
                if l == 0:
                    to_merge = [row_2_1, row_2_2, row_2_3]
                else:
                    merge_component(to_merge[1], row_2_2, l)
                    merge_component(to_merge[2], row_2_3, l)
            row_2_1, row_2_2, row_2_3 = to_merge

            row_3_1 = copy.deepcopy(start_node)
            row_3_1.resample(True)
            for l in range(len(rule_groups)):
                rule_group = rule_groups[l]
                rule_num_pos = rule_group[0]
                row_3_2 = rule_num_pos.apply_rule(row_3_1)
                row_3_3 = rule_num_pos.apply_rule(row_3_2)
                for i in range(1, len(rule_group)):
                    rule = rule_group[i]
                    row_3_2 = rule.apply_rule(row_3_1, row_3_2)
                for i in range(1, len(rule_group)):
                    rule = rule_group[i]
                    row_3_3 = rule.apply_rule(row_3_2, row_3_3)
                if l == 0:
                    to_merge = [row_3_1, row_3_2, row_3_3]
                else:
                    merge_component(to_merge[1], row_3_2, l)
                    merge_component(to_merge[2], row_3_3, l)
            row_3_1, row_3_2, row_3_3 = to_merge

            imgs = [render_panel(row_1_1),
                    render_panel(row_1_2),
                    render_panel(row_1_3),
                    render_panel(row_2_1),
                    render_panel(row_2_2),
                    render_panel(row_2_3),
                    render_panel(row_3_1),
                    render_panel(row_3_2),
                    np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)]
            context = [row_1_1, row_1_2, row_1_3, row_2_1, row_2_2, row_2_3, row_3_1, row_3_2]
            
            answer_AoT = copy.deepcopy(row_3_3)
            modifiable_attr = sample_attr_avail(rule_groups, answer_AoT)

            if not args.fair:
                candidates, answers_imgs = original_raven(modifiable_attr, answer_AoT, rule_groups, context)
            else:
                candidates, answers_imgs = fair_raven(modifiable_attr, answer_AoT, rule_groups, context)

            if args.save:
                zipped = list(zip(candidates, answers_imgs))
                random.shuffle(zipped)
                candidates, answers_imgs = zip(*zipped)

                image = imgs[0:8] + list(answers_imgs)
                target = candidates.index(answer_AoT)
                _, predicted = solve(rule_groups, context, candidates)
                meta_matrix, meta_target = serialize_rules(rule_groups)
                structure, meta_structure = serialize_aot(start_node)
                np.savez_compressed("{}/{}/RAVEN_{}_{}.npz".format(args.save_dir, key, k, set_name), image=image,
                                    target=target,
                                    predict=predicted,
                                    meta_matrix=meta_matrix,
                                    meta_target=meta_target,
                                    structure=structure,
                                    meta_structure=meta_structure)

                with open("{}/{}/RAVEN_{}_{}.xml".format(args.save_dir, key, k, set_name), "w") as f:
                    dom = dom_problem(context + list(candidates), rule_groups)
                    f.write(dom)


def original_raven(modifiable_attr, answer_AoT, rule_groups, context):
    candidates = [answer_AoT]
    answers_imgs = [render_panel(answer_AoT)]

    answer_score, _ = solve(rule_groups, context, [answer_AoT])
    assert answer_score > 0

    """Create the negative choices for the original RAVEN dataset"""
    while len(candidates) < 8:
        component_idx, attr_name, min_level, max_level = sample_attr(modifiable_attr)
        new_answer = copy.deepcopy(answer_AoT)
        new_answer.sample_new(component_idx, attr_name, min_level, max_level, answer_AoT)

        new_answer_img = render_panel(new_answer)
        ok = True
        new_answer_score, _ = solve(rule_groups, context, [new_answer])
        if new_answer_score >= answer_score:
            print 'Warning - Accidentally generated good answer - resampling'
            ok = False
        for i in range(0, len(answers_imgs)):
            if (new_answer_img == answers_imgs[i]).all():
                print 'Warning - New answer equals existing image - resampling'
                ok = False

        if ok:
            candidates.append(new_answer)
            answers_imgs.append(new_answer_img)

    return candidates, answers_imgs


def fair_raven(modifiable_attr, answer_AoT, rule_groups, context):
    candidates = [answer_AoT]
    answers_imgs = [render_panel(answer_AoT)]

    answer_score, _ = solve(rule_groups, context, [answer_AoT])
    assert answer_score > 0

    """Create the negative choices for the balanced RAVEN-FAIR dataset"""
    attrs = [modifiable_attr]
    idxs = []
    blacklist = [[]]

    try:
        while len(candidates) < 8:
            while True:
                indices = random.sample(range(len(candidates)), k=len(candidates))
                timeout_flag = False
                for idx in indices:
                    if len(attrs[idx]) > 0:
                        timeout_flag = True
                        break
                if timeout_flag:
                    break
                    print 'No option to continue'
                raise Exception('No option to continue')

            attr_i = attrs[idx]
            candidate_i = candidates[idx]
            blacklist_i = blacklist[idx]

            component_idx, attr_name, min_level, max_level = sample_attr(attr_i)
            try:
                with timeout(5):
                    new_answer = copy.deepcopy(candidate_i)
                    new_answer.sample_new(component_idx, attr_name, min_level, max_level, candidate_i)
                    new_attr = sample_attr_avail(rule_groups, new_answer)
            except Exception as e:
                print 'Attempt to sample failed - recovering'
                print(e)
                print(idxs)
                print(component_idx, attr_name, min_level, max_level)
                for attr in attr_i:
                    print(attr)
                print(blacklist_i)
                continue

            new_blacklist = copy.deepcopy(blacklist_i) + [attr_name]
            for i in reversed(range(len(new_attr))):
                if new_attr[i][1] in new_blacklist:
                    new_attr.pop(i)

            new_answer_img = render_panel(new_answer)
            ok = True
            new_answer_score, _ = solve(rule_groups, context, [new_answer])
            if new_answer_score >= answer_score:
                print 'Warning - Accidentally generated good answer - resampling'
                ok = False
            for i in range(0, len(answers_imgs)):
                if (new_answer_img == answers_imgs[i]).all():
                    print 'Warning - New answer equals existing image - resampling'
                    ok = False
            if ok:
                idxs.append(idx)
                candidates.append(new_answer)
                attrs.append(new_attr)
                blacklist.append(new_blacklist)
                answers_imgs.append(new_answer_img)

    except Exception as e:
        print(e)
        raise e

    return candidates, answers_imgs


def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for RAVEN")
    main_arg_parser.add_argument("--num-samples", type=int, default=10000,
                                 help="number of samples for each component configuration")
    main_arg_parser.add_argument("--save-dir", type=str, default="./Datasets",
                                 help="path to folder where the generated dataset will be saved.")
    main_arg_parser.add_argument("--seed", type=int, default=1234,
                                 help="random seed for dataset generation")
    main_arg_parser.add_argument("--fair", type=int, default=0,
                                 help="whether to create FAIR or ORIG dataset")
    main_arg_parser.add_argument("--val", type=float, default=2,
                                 help="the proportion of the size of validation set")
    main_arg_parser.add_argument("--test", type=float, default=2,
                                 help="the proportion of the size of test set")
    main_arg_parser.add_argument("--save", type=int, default=1,
                                 help="save the dataset")
    args = main_arg_parser.parse_args()

    args.save_dir = os.path.join(args.save_dir, 'RAVEN' + ('-F' if args.fair else ''))

    all_configs = {"center_single": build_center_single(),
                   "distribute_four": build_distribute_four(),
                   "distribute_nine": build_distribute_nine(),
                   "left_center_single_right_center_single": build_left_center_single_right_center_single(),
                   "up_center_single_down_center_single": build_up_center_single_down_center_single(),
                   "in_center_single_out_center_single": build_in_center_single_out_center_single(),
                   "in_distribute_four_out_center_single": build_in_distribute_four_out_center_single()}

    separate(args, all_configs)


if __name__ == "__main__":
    main()
