import os
import time
import keras
import numpy as np
import tensorflow as tf
import mlflow
import psutil
import pynvml
from mlflow.keras import autolog
from model.layer import TransformerBlock
from keras.callbacks import TensorBoard

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
TB_LOG_DIR  = os.path.join(BASE_DIR, "logs")
MLFLOW_DIR  = os.path.join(BASE_DIR, "mlruns")
os.makedirs(TB_LOG_DIR, exist_ok=True)
os.makedirs(MLFLOW_DIR, exist_ok=True)

mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
mlflow.set_experiment("japan_cooperation _experiment")
autolog(log_models=True, log_input_examples=True, log_model_signatures=True)

pynvml.nvmlInit() 
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0) 

BATCHSIZE = 16
LRBASE = 4E-6
WEIGHTDECAY = 0.9
EPOCHS = 5
SEQ_LEN = 128           
D_MODEL = 64
REG_TARGET_DIM = 3

N_TRAIN = 1000
N_VAL   = 200

def build_model():
    inputs = keras.Input(shape=(SEQ_LEN, D_MODEL), name="input_sequence")
    x = TransformerBlock(
        out_dim=D_MODEL,
        num_heads=4,
        head_dim=64,
        mlp_rate=4,
        dropout_rate=0.1
    )(inputs)
    x = keras.layers.GlobalAveragePooling1D()(x)
    outputs = keras.layers.Dense(REG_TARGET_DIM, activation=None, name="regressor")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="TransformerRegressor")

class SystemMetricsCallback(tf.keras.callbacks.Callback):
    """
    At the end of each epoch, this callback:
      - measures GPU utilization, memory, temperature, power, fan speed, fragmentation
      - measures CPU utilization, RAM used/free, disk usage
      - measures disk I/O and network I/O deltas since epoch start
      - logs all metrics to both TensorBoard and MLflow
      - logs model size on disk and total parameter count
    """
    def __init__(self, tensorboard_log_dir):
        super().__init__()
        self.tb_writer = tf.summary.create_file_writer(tensorboard_log_dir)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        self.disk_io_start = psutil.disk_io_counters()
        self.net_io_start = psutil.net_io_counters()

    def on_epoch_end(self, epoch, logs=None):
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - self.epoch_start_time

        gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
        gpu_memory_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
        gpu_temperature = pynvml.nvmlDeviceGetTemperature(gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
        gpu_power_watts = pynvml.nvmlDeviceGetPowerUsage(gpu_handle) / 1000.0
        gpu_fan_percent = pynvml.nvmlDeviceGetFanSpeed(gpu_handle)
        gpu_mem_fragmentation = (gpu_memory_info.free / gpu_memory_info.total * 100.0)
        gpu_peak_mem_mb = gpu_memory_info.used / (1024**2)

        cpu_util_percent = psutil.cpu_percent()
        virtual_mem = psutil.virtual_memory()
        disk_usage_percent = psutil.disk_usage(BASE_DIR).percent

        disk_io_end = psutil.disk_io_counters()
        disk_read_mb = (disk_io_end.read_bytes - self.disk_io_start.read_bytes) / (1024**2)
        disk_write_mb = (disk_io_end.write_bytes - self.disk_io_start.write_bytes) / (1024**2)
        net_io_end = psutil.net_io_counters()
        net_sent_mb = (net_io_end.bytes_sent - self.net_io_start.bytes_sent) / (1024**2)
        net_recv_mb = (net_io_end.bytes_recv - self.net_io_start.bytes_recv) / (1024**2)

        total_parameters = self.model.count_params()
        model_filepath = os.path.join(BASE_DIR, 'final_model.keras')
        model_size_mb = (os.path.getsize(model_filepath) / 
                         (1024**2)if os.path.exists(model_filepath) else 0)
        
        metrics = {
            'gpu_util_percent': gpu_utilization,
            'gpu_mem_used_mb': gpu_memory_info.used / (1024**2),
            'gpu_mem_free_mb': gpu_memory_info.free / (1024**2),
            'gpu_temp_c': gpu_temperature,
            'gpu_power_watts': gpu_power_watts,
            'gpu_fan_percent': gpu_fan_percent,
            'gpu_fragmentation_percent': gpu_mem_fragmentation,
            'gpu_peak_mem_mb': gpu_peak_mem_mb,
            'cpu_util_percent': cpu_util_percent,
            'ram_used_mb': virtual_mem.used / (1024**2),
            'ram_free_mb': virtual_mem.available / (1024**2),
            'disk_usage_percent': disk_usage_percent,
            'disk_read_mb': disk_read_mb,
            'disk_write_mb': disk_write_mb,
            'net_sent_mb': net_sent_mb,
            'net_recv_mb': net_recv_mb,
            'epoch_duration_s': epoch_duration,
            'model_size_mb': model_size_mb,
            'num_parameters': total_parameters,
        }

        if logs:
            for name, value in logs.items():
                metrics[name] = float(value)

        with self.tb_writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar(name, value, step=epoch)
            self.tb_writer.flush()

        for name, value in metrics.items():
            mlflow.log_metric(name, value, step=epoch)

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0" # set on specific GPU, Default 3070
    gpus = tf.config.list_physical_devices(device_type='GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, enable=True)
            # tf.config.set_logical_device_configuration(gpu,
            # [tf.config.LogicalDeviceConfiguration(memory_limit=8192)])
    
    boundaries = [int(EPOCHS * 0.2), int(EPOCHS * 0.4), int(EPOCHS * 0.6)]
    values = [v * LRBASE for v in [1.0, 0.8, 0.7, 0.4]]

    lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, values
    )

    optimizer = keras.optimizers.Lion(learning_rate=lr_schedule, weight_decay=WEIGHTDECAY)
    
    callbacks = [
        TensorBoard(log_dir=TB_LOG_DIR, histogram_freq=1, update_freq='epoch'),
        keras.callbacks.ModelCheckpoint("checkpoint.keras", monitor="val_loss", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=20),
        SystemMetricsCallback(TB_LOG_DIR)
    ]

    model = build_model()
    model.compile(
        optimizer=optimizer, loss="mse", metrics=["mae"])

    x_train = np.random.rand(N_TRAIN, SEQ_LEN, D_MODEL).astype(np.float32)
    y_train = np.random.rand(N_TRAIN, REG_TARGET_DIM).astype(np.float32)
    x_val   = np.random.rand(N_VAL,   SEQ_LEN, D_MODEL).astype(np.float32)
    y_val   = np.random.rand(N_VAL,   REG_TARGET_DIM).astype(np.float32)
    
    with mlflow.start_run():
        mlflow.log_params({
            "batch_size": BATCHSIZE,
            "base_lr": LRBASE,
            "weight_decay": WEIGHTDECAY,
            "epochs": EPOCHS
        })

        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            batch_size=BATCHSIZE,
            epochs=EPOCHS,
            callbacks=callbacks,
        )
        model.save("final_model.keras")                     