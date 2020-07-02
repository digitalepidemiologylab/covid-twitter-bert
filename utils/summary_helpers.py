import os
import tensorflow as tf
#from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorflow.python.framework import tensor_util
from google.cloud import storage

# Reads the summary files created by tensor
# If no tag is specified it returns a set with available tags 
def read_summary_file(path_to_events_file, tag = ""):
  valuesDict = dict()
  tagSet = set()
  event_files = []

  #If a directory instead of the exact filename is given, look up the first events file
  if ".tfevents." not in path_to_events_file:
    if 'gs://' in path_to_events_file:
      event_files = tf.io.gfile.listdir(path_to_events_file)
    else:
      event_files = os.listdir(path_to_events_file)

    if not len(event_files):
      raise Exception("No valid event files found in this location")
    elif len(event_files) > 1:
      import warnings
      warnings.warn("Warning: There are multiple event files in the directory. Reading the first one! If you need to read another one, please specify the exact filename you need to read instead. ")

    path_to_events_file = os.path.join(path_to_events_file,event_files[0])
  
  #for e in summary_iterator(path_to_events_file): 
  for e in tf.data.TFRecordDatase(path_to_events_file):
      for v in e.summary.value:
      tagSet.add(v.tag)
      if v.tag == tag:
        valuesDict.update( {int(e.step) : float(tensor_util.MakeNdarray(v.tensor))} )

  if tag:
    return valuesDict
  else: 
    return tagSet


# Helper function specific for the CovidBert-project.
# If bucket is not specified it reads in the local synced directory
def cb_read_summary_file(run_name, tag = None, bucket = None, dataset = "eval", project_name="covid-bert-v2", train_level="pretrain"):
  path_to_events_file = os.path.join(project_name,train_level,"runs",run_name,'summaries',dataset)

  if bucket:
    path_to_events_file = os.path.join("gs://", bucket, path_to_events_file)

  output = read_summary_file(path_to_events_file, tag)
  
  return output
