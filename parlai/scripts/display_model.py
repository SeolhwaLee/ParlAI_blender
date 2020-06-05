#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Basic example which iterates through the tasks specified and runs the given model on
them.

Examples
--------

.. code-block:: shell

  python examples/display_model.py -t babi:task1k:1 -m "repeat_label"
  python examples/display_model.py -t "#MovieDD-Reddit" -m "ir_baseline" -mp "-lp 0.5" -dt test
"""  # noqa: E501
import time

from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.scripts.script import ParlaiScript
from parlai.utils.strings import colorize

import random


def simple_display(opt, world, turn):
    if opt['batchsize'] > 1:
        raise RuntimeError('Simple view only support batchsize=1')
    teacher, response = world.get_acts()
    if turn == 0:
        text = "- - - NEW EPISODE: " + teacher.get('id', "[no agent id]") + "- - -"
        print(colorize(text, 'highlight'))
    text = teacher.get('text', '[no text field]')
    print(colorize(text, 'text'))
    response_text = response.get('text', 'No response')
    labels = teacher.get('labels', teacher.get('eval_labels', ['[no labels field]']))
    labels = '|'.join(labels)
    print(colorize('    labels: ' + labels, 'labels'))
    print(colorize('     model: ' + response_text, 'text2'))




def simple_display_chateval(opt, world, turn):
    if opt['batchsize'] > 1:
        raise RuntimeError('Simple view only support batchsize=1')
    teacher, response = world.get_acts()
    if turn == 0:
        text = "- - - NEW EPISODE: " + teacher.get('id', "[no agent id]") + "- - -"
        print(colorize(text, 'highlight'))
    text = teacher.get('text', '[no text field]')
    print(colorize(text, 'text'))
    response_text = response.get('text', 'No response')
    labels = teacher.get('labels', teacher.get('eval_labels', ['[no labels field]']))
    labels = '|'.join(labels)
    print(colorize('    labels: ' + labels, 'labels'))
    # print(colorize('     model: ' + response_text, 'text2'))

    # for the chateval
    timestr = time.strftime("%Y%m%d-%H%M%S")
    # script_input_path = str(opt.get('script_input_path'))
    # script_file = open(script_input_path, 'r', encoding='utf-8')
    # file_name = script_input_path.split('/')[-1].split('.')[0]

    if opt.get('chateval_multi') == True:
        print(colorize('     model: ' + response_text, 'text2'))
        script_response = open(opt.get('script_out_path') + '/' + 'test' + '_' + opt.get('model_file') + '_' + timestr +
                               '.txt', 'w')
        print(colorize('     model: ' + response_text, 'text2'))

        script_response.write("%s\n" % (response_text))

        # if opt.get('chateval_multi_num') == 2:
        #     script_response.write("%s\n" % (response_text))
        # elif opt.get('chateval_multi_num') == 3:
        #     pass
        # else:
        #     print("We only consider to 2 and 3 turns")
    else:
        print("Chateval multiturn script something wrong!")


def setup_args():
    parser = ParlaiParser(True, True, 'Display model predictions.')
    parser.add_argument('-n', '-ne', '--num-examples', default=10)
    parser.add_argument('--display-ignore-fields', type=str, default='')
    parser.add_argument(
        '--verbose',
        type='bool',
        default=False,
        hidden=True,
        help='Display additional debug info, e.g. the per-token loss breakdown for generative models.',
    )
    parser.add_argument(
        '--chateval-multi',
        type='bool',
        default=False,
        hidden=True,
        dest='chateval_multi',
        help='True is chateval multiturn setting, False just single turn.',
    )
    parser.add_argument(
        '--chateval-multi-num',
        type=int,
        default=0,
        dest='chateval_multi_num',
        help='True is chateval multiturn setting, turn coverage count.',
    )
    parser.add_argument(
        '-scip',
        '--chateval-input-path',
        type=str,
        default=None,
        dest='script_input_path',
        help='Chateval script input path',
    )
    parser.add_argument(
        '-scop',
        '--chateval-output-path',
        type=str,
        default=None,
        dest='script_output_path',
        help='Chateval result output path',
    )
    # by default we want to display info about the validation set
    parser.set_defaults(datatype='valid')
    return parser


def display_model(opt):
    random.seed(42)

    # Create model and assign it to the specified task
    agent = create_agent(opt)
    world = create_task(opt, agent)

    # Show some example dialogs.
    # turn = 0
    # with world:
    #     for _k in range(int(opt['num_examples'])):
    #         world.parley()
    #         if opt['verbose']:
    #             print(world.display() + "\n~~")
    #         else:
    #             simple_display(opt, world, turn)
    #         turn += 1
    #         if world.get_acts()[0]['episode_done']:
    #             turn = 0
    #         if world.epoch_done():
    #             print("EPOCH DONE")
    #             turn = 0
    #             break


    if opt.get('chateval_multi') == False:
        # Show some example dialogs.
        turn = 0
        with world:
            for _k in range(int(opt['num_examples'])):
                world.parley()
                if opt['verbose']:
                    print(world.display() + "\n~~")
                else:
                    simple_display(opt, world, turn)
                turn += 1
                if world.get_acts()[0]['episode_done']:
                    turn = 0
                if world.epoch_done():
                    print("EPOCH DONE")
                    turn = 0
                    break
    else:
        # Show some example dialogs.
        turn = 0
        with world:
            for _k in range(int(opt['num_examples'])):
                world.parley()
                if opt['verbose']:
                    print(world.display() + "\n~~")
                else:
                    simple_display_chateval(opt, world, turn)
                turn += 1
                if world.get_acts()[0]['episode_done']:
                    turn = 0
                if world.epoch_done():
                    print("EPOCH DONE")
                    turn = 0
                    break

class DisplayModel(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        display_model(self.opt)


if __name__ == '__main__':
    DisplayModel.main()
