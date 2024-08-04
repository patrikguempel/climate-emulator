import tensorflow as tf
import os

def set_environment(workers_per_node, workers_per_gpu):
    print('<< set_environment START >>')
    nodename = os.environ['SLURMD_NODENAME']
    procid = os.environ['SLURM_LOCALID']
    print(f'node name: {nodename}')
    print(f'procid:    {procid}')
    # stream = os.popen('scontrol show hostname $SLURM_NODELIST')
    # output = stream.read()
    # oracle = output.split("\n")[0]
    if procid==str(workers_per_node): # This takes advantage of the fact that procid numbering starts with ZERO
        os.environ["KERASTUNER_TUNER_ID"] = "chief"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("Keras Tuner Oracle has been assigned.")
    else:
        os.environ["KERASTUNER_TUNER_ID"] = "tuner-" + str(nodename) + "-" + str(procid)
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{int(int(procid)//workers_per_gpu)}"
    print(f'SY DEBUG: procid-{procid} / GPU-ID-{os.environ["CUDA_VISIBLE_DEVICES"]}')

    #print(os.environ)
    print('<< set_environment END >>')


set_environment(1, 1)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))