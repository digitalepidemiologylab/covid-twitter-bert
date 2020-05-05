import tensorflow as tf

def main():
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu='node-1',
            zone='europe-west4-a',
            project='vacma-250010')
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
    __import__('pdb').set_trace()

if __name__ == "__main__":
    main()
