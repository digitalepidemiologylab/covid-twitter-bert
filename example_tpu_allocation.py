import datetime
from utils.misc import create_TPU, destroy_TPU



ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')

tpu_name = f"auto-preempt-{ts}"
zone = "us-central1-f" 

tpu_ip = create_TPU(tpu_name,zone)
destroy_TPU(tpu_name)






