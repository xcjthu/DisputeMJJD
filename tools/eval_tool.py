import logging
import os
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from timeit import default_timer as timer

logger = logging.getLogger(__name__)


def gen_time_str(t):
    t = int(t)
    minute = t // 60
    second = t % 60
    return '%2d:%02d' % (minute, second)

import code, traceback, signal

def debug(sig, frame):
    """Interrupt running process, and provide a python prompt for
    interactive debugging."""
    d={'_frame':frame}         # Allow access to frame object.
    d.update(frame.f_globals)  # Unless shadowed by global
    d.update(frame.f_locals)
    message  = "Signal received : entering python shell.\nTraceback:\n"
    message += ''.join(traceback.format_stack(frame))
    print(message)

def listen():
    signal.signal(signal.SIGUSR1, debug)  # Register handler


def output_value(epoch, mode, step, time, loss, info, end, config):
    try:
        delimiter = config.get("output", "delimiter")
    except Exception as e:
        delimiter = " "
    s = ""
    s = s + str(epoch) + " "
    while len(s) < 7:
        s += " "
    s = s + str(mode) + " "
    while len(s) < 14:
        s += " "
    s = s + str(step) + " "
    while len(s) < 25:
        s += " "
    s += str(time)
    while len(s) < 40:
        s += " "
    s += str(loss)
    while len(s) < 48:
        s += " "
    s += str(info)
    s = s.replace(" ", delimiter)
    if not (end is None):
        print(s, end=end)
    else:
        print(s)


def valid(model, dataset, epoch, writer, config, gpu_list, output_function, mode="valid"):
    listen()
    model.eval()
    local_rank = config.getint('distributed', 'local_rank')

    acc_result = None
    total_loss = 0
    cnt = 0
    total_len = len(dataset)
    start_time = timer()
    output_info = ""

    output_time = config.getint("output", "output_time")
    step = -1
    more = ""
    if total_len < 10000:
        more = "\t"

    for step, data in enumerate(dataset):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if len(gpu_list) > 0:
                    data[key] = Variable(data[key].cuda())
                else:
                    data[key] = Variable(data[key])
        # print(data.keys())
        results = model(data, config, gpu_list, acc_result, "valid")

        loss, acc_result = results["loss"], results["acc_result"]
        total_loss += float(loss)
        cnt += 1
        # break

        if step % output_time == 0 and local_rank <= 0:
            delta_t = timer() - start_time

            output_value(epoch, mode, "%d/%d" % (step + 1, total_len), "%s/%s" % (
                gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                         "%.3lf" % (total_loss / (step + 1)), output_info, '\r', config)
    '''
    if step == -1:
        logger.error("There is no data given to the model in this epoch, check your data.")
        raise NotImplementedError()
    print("before gather", local_rank)
    if config.getboolean("distributed", "use"):
        if type(acc_result) == list:
            mytensor = torch.LongTensor([cl[key] for cl in acc_result for key in cl]).to(gpu_list[local_rank])
        else:
            mytensor = torch.LongTensor([acc_result[key] for key in acc_result]).to(gpu_list[local_rank])
        shape = mytensor.shape
        # print(mytensor)
        mylist = [torch.LongTensor(shape).to(gpu_list[local_rank]) for i in range(config.getint('distributed', 'gpu_num'))]
        torch.distributed.all_gather(mylist, mytensor)#, 0)
        print("after gather", local_rank)
        if local_rank == 0:
            print(mylist)
        if local_rank == 0:
            mytensor = sum(mylist)
            # print(mytensor)
            index = 0
            if type(acc_result) == list:
                ind = 0
                for i in range(len(acc_result)):
                    for key in acc_result[i].keys():
                        acc_result[i][key] = int(mytensor[ind])
                        ind += 1
            else:
                for key in acc_result:
                    acc_result[key] = int(mytensor[index])
                    index += 1
    '''
    # if local_rank <= 0:
    if True:
        delta_t = timer() - start_time
        print("before output")
        output_info = output_function(acc_result, config)
        print("after output")
        output_value(epoch, mode, "%d/%d" % (step + 1, total_len), "%s/%s" % (
            gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                    "%.3lf" % (total_loss / (step + 1)), output_info, None, config)

        writer.add_scalar(config.get("output", "model_name") + "_eval_epoch", float(total_loss) / (step + 1),
                        epoch)
    torch.distributed.barrier()
    # model.train()
