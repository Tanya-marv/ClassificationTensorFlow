"""This example shows how to train an image classifier based on any
TensorFlow Hub module that computes image feature vectors. By default,
it uses the feature vectors computed by Inception V3 trained on ImageNet.
For more options, search https://tfhub.dev for image feature vector modules.

The top layer receives as input a 2048-dimensional vector (assuming
Inception V3) for each image. We train a softmax layer on top of this
representation. If the softmax layer contains N labels, this corresponds
to learning N + 2048*N model parameters for the biases and weights.

image sourse -> http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar

Here's an example, which assumes you have a folder containing class-named
subfolders, each full of images for each label.

The subfolder names are important, since they define what label is applied to
each image, but the filenames themselves don't matter.

You can replace the image_dir argument with any folder containing subfolders of
images. The label for each image is taken from the name of the subfolder it's
in.

By default this script will use the highly accurate, but comparatively large and
slow Inception V3 model architecture. It's recommended that you start with this
to validate that you have gathered good training data, but if you want to deploy
on resource-limited platforms, you can try the `--tfhub_module` flag with a
Mobilenet model.

These instrumented models can be converted to fully quantized mobile models via
TensorFlow Lite.

There are different Mobilenet models to choose from, with a variety of file
size and latency options.
  - The first number can be '100', '075', '050', or '025' to control the number
    of neurons; the number of weights shrinks with the square of that
    fraction.
  - The second number is the input image size. You can choose '224', '192',
    '160', or '128', with smaller sizes giving faster speeds.
Visualize the summaries with this command:

tensorboard --logdir=training_summaries

To use with Tensorflow Serving, run this tool with --saved_model_dir set
to some increasingly numbered export location under the model base path, e.g.:
python3 -m retrain \
--bottleneck_dir=bottlenecks \
--how_many_training_steps=1000 \
--model_dir=models/ \
--summaries_dir=training_summaries/ \
--output_graph=retrained.pb \
--output_labels=labels.txt \
--image_dir=images/Images
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import argparse
from datetime import datetime
import sys

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.quantization import quantize as contrib_quantize

import visualization_utils as visual
import image_utils as img_utils
import system_utils
import bottleneck
import logger as log

FLAGS = None

def add_final_retrain_ops(class_count, final_tensor_name, bottleneck_tensor,
                          quantize_layer, is_training):
  """Adds a new softmax and fully-connected layer for training and eval.

  We need to retrain the top layer to identify our new classes, so this function
  adds the right operations to the graph, along with some variables to hold the
  weights, and then sets up all the gradients for the backward pass.

  The set up for the softmax and fully-connected layers is based on:
  https://www.tensorflow.org/tutorials/mnist/beginners/index.html

  Args:
    class_count: Integer of how many categories of things we're trying to
        recognize.
    final_tensor_name: Name string for the new final node that produces results.
    bottleneck_tensor: The output of the main CNN graph.
    quantize_layer: Boolean, specifying whether the newly added layer should be
        instrumented for quantization with TF-Lite.
    is_training: Boolean, specifying whether the newly add layer is for training
        or eval.

  Returns:
    The tensors for the training and cross entropy results, and tensors for the
    bottleneck input and ground truth input.
  """
    batch_size, bottleneck_tensor_size = bottleneck_tensor.get_shape().as_list()
    assert batch_size is None, 'We want to work with arbitrary batch size.'
    with tf.name_scope('input'):
        bottleneck_input = tf.placeholder_with_default(
            bottleneck_tensor,
            shape=[batch_size, bottleneck_tensor_size],
            name='BottleneckInputPlaceholder')

        ground_truth_input = tf.placeholder(
            tf.int64, [batch_size], name='GroundTruthInput')

    # Organizing the following ops so they are easier to see in TensorBoard.
    layer_name = 'final_retrain_ops'
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            initial_value = tf.truncated_normal(
                [bottleneck_tensor_size, class_count], stddev=0.001)
            layer_weights = tf.Variable(initial_value, name='final_weights')
            visual.variable_summaries(layer_weights)

        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
            visual.variable_summaries(layer_biases)

        with tf.name_scope('Wx_plus_b'):
            logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
            tf.summary.histogram('pre_activations', logits)

    final_tensor = tf.nn.softmax(logits, name=final_tensor_name)

    # The tf.contrib.quantize functions rewrite the graph in place for
    # quantization. The imported model graph has already been rewritten, so upon
    # calling these rewrites, only the newly added final layer will be
    # transformed.
    if quantize_layer:
        if is_training:
            contrib_quantize.create_training_graph()
        else:
            contrib_quantize.create_eval_graph()

    tf.summary.histogram('activations', final_tensor)

    # If this is an eval graph, we don't need to add loss ops or an optimizer.
    if not is_training:
        return None, None, bottleneck_input, ground_truth_input, final_tensor

    with tf.name_scope('cross_entropy'):
        cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
            labels=ground_truth_input, logits=logits)

    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        train_step = optimizer.minimize(cross_entropy_mean)

    return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
            final_tensor)

def add_evaluation_step(result_tensor, ground_truth_tensor):
    """Inserts the operations we need to evaluate the accuracy of our results.

  Args:
    result_tensor: The new final node that produces results.
    ground_truth_tensor: The node we feed ground truth data
    into.

  Returns:
    Tuple of (evaluation step, prediction).
  """
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            prediction = tf.argmax(result_tensor, 1)
            correct_prediction = tf.equal(prediction, ground_truth_tensor)
        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)
    return evaluation_step, prediction

def run_final_eval(train_session, module_spec, class_count, image_lists,
                   jpeg_data_tensor, decoded_image_tensor,
                   resized_image_tensor, bottleneck_tensor):
    """Runs a final evaluation on an eval graph using the test data set.

  Args:
    train_session: Session for the train graph with the tensors below.
    module_spec: The hub.ModuleSpec for the image module being used.
    class_count: Number of classes
    image_lists: OrderedDict of training images for each label.
    jpeg_data_tensor: The layer to feed jpeg image data into.
    decoded_image_tensor: The output of decoding and resizing the image.
    resized_image_tensor: The input node of the recognition graph.
    bottleneck_tensor: The bottleneck output layer of the CNN graph.
  """
    test_bottlenecks, test_ground_truth, test_filenames = (
        bottleneck.get_random_cached_bottlenecks(train_session, image_lists,
                                                 FLAGS.test_batch_size,
                                      'testing', FLAGS.bottleneck_dir,
                                                 FLAGS.image_dir, jpeg_data_tensor,
                                                 decoded_image_tensor, resized_image_tensor,
                                                 bottleneck_tensor, FLAGS.tfhub_module))

    (eval_session, _, bottleneck_input, ground_truth_input, evaluation_step,
     prediction) = build_eval_session(module_spec, class_count)
    test_accuracy, predictions = eval_session.run(
        [evaluation_step, prediction],
        feed_dict={
            bottleneck_input: test_bottlenecks,
            ground_truth_input: test_ground_truth
        })
    logging.info('Final test accuracy = %.1f%% (N=%d)',
                 test_accuracy * 100, len(test_bottlenecks))

    if FLAGS.print_misclassified_test_images:
        logging.info('=== MISCLASSIFIED TEST IMAGES ===')
        for i, test_filename in enumerate(test_filenames):
            if predictions[i] != test_ground_truth[i]:
                logging.info('%70s  %s', test_filename,
                             list(image_lists.keys())[predictions[i]])

def build_eval_session(module_spec, class_count):
    """Builds an restored eval session without train operations for exporting.

  Args:
    module_spec: The hub.ModuleSpec for the image module being used.
    class_count: Number of classes

  Returns:
    Eval session containing the restored eval graph.
    The bottleneck input, ground truth, eval step, and prediction tensors.
  """
    # If quantized, we need to create the correct eval graph for exporting.
    eval_graph, bottleneck_tensor, resized_input_tensor, wants_quantization = (
        bottleneck.create_module_graph(module_spec))

    eval_sess = tf.Session(graph=eval_graph)
    with eval_graph.as_default():
        # Add the new layer for exporting.
        (_, _, bottleneck_input,
         ground_truth_input, final_tensor) = add_final_retrain_ops(
            class_count, FLAGS.final_tensor_name, bottleneck_tensor,
            wants_quantization, is_training=False)

        # Now we need to restore the values from the training graph to the eval
        # graph.
        tf.train.Saver().restore(eval_sess, FLAGS.checkpoint_path)

        evaluation_step, prediction = add_evaluation_step(final_tensor,
                                                          ground_truth_input)

    return (eval_sess, resized_input_tensor, bottleneck_input, ground_truth_input,
            evaluation_step, prediction)

def save_graph_to_file(graph_file_name, module_spec, class_count):
    """Saves an graph to file, creating a valid quantized one if necessary."""
    sess, _, _, _, _, _ = build_eval_session(module_spec, class_count)
    graph = sess.graph

    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [FLAGS.final_tensor_name])

    with tf.gfile.GFile(graph_file_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())

def export_model(module_spec, class_count, saved_model_dir):
    """Exports model for serving.

  Args:
    module_spec: The hub.ModuleSpec for the image module being used.
    class_count: The number of classes.
    saved_model_dir: Directory in which to save exported model and variables.
  """
    # The SavedModel should hold the eval graph.
    sess, in_image, _, _, _, _ = build_eval_session(module_spec, class_count)
    with sess.graph.as_default() as graph:
        tf.saved_model.simple_save(
            sess,
            saved_model_dir,
            inputs={'image': in_image},
            outputs={'prediction': graph.get_tensor_by_name('final_result:0')},
            legacy_init_op=tf.group(tf.tables_initializer(), name='legacy_init_op')
        )


def main(_):
    log.initialize_log_file()
    logging_verbosity = log.logging_level_verbosity(FLAGS.logging_verbosity)
    logging.set_verbosity(logging_verbosity)

    logging.error('WARNING: This tool is deprecated in favor of '
                  'https://github.com/tensorflow/hub/tree/master/'
                  'tensorflow_hub/tools/make_image_classifier')

    if not FLAGS.image_dir:
        logging.error('Must set flag --image_dir.')
        return -1

    # Prepare necessary directories that can be used during training
    system_utils.prepare_file_system(FLAGS)

    # Look at the folder structure, and create lists of all the images.
    image_lists = img_utils.create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage,
                                     FLAGS.validation_percentage)
    class_count = len(image_lists.keys())
    if class_count == 0:
        logging.error('No valid folders of images found at %s', FLAGS.image_dir)
        return -1
    if class_count == 1:
        logging.error('Only one valid folder of images found at %s '
                      ' - multiple classes are needed for classification.',
                      FLAGS.image_dir)
        return -1

    # See if the command-line flags mean we're applying any distortions.
    do_distort_images = img_utils.should_distort_images(
        FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
        FLAGS.random_brightness)

    # Set up the pre-trained graph.
    module_spec = hub.load_module_spec(FLAGS.tfhub_module)
    graph, bottleneck_tensor, resized_image_tensor, wants_quantization = (
        bottleneck.create_module_graph(module_spec))

    # Add the new layer that we'll be training.
    with graph.as_default():
        (train_step, cross_entropy, bottleneck_input,
         ground_truth_input, final_tensor) = add_final_retrain_ops(
            class_count, FLAGS.final_tensor_name, bottleneck_tensor,
            wants_quantization, is_training=True)

    with tf.Session(graph=graph) as sess:
        # Initialize all weights: for the module to their pretrained values,
        # and for the newly added retraining layer to random initial values.
        init = tf.global_variables_initializer()
        sess.run(init)

        # Set up the image decoding sub-graph.
        jpeg_data_tensor, decoded_image_tensor = img_utils.add_jpeg_decoding(module_spec)

        if do_distort_images:
            # We will be applying distortions, so set up the operations we'll need.
            (distorted_jpeg_data_tensor,
             distorted_image_tensor) = img_utils.add_input_distortions(
                FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
                FLAGS.random_brightness, module_spec)
        else:
            # We'll make sure we've calculated the 'bottleneck' image summaries and
            # cached them on disk.
            bottleneck.cache_bottlenecks(sess, image_lists, FLAGS.image_dir,
                                         FLAGS.bottleneck_dir, jpeg_data_tensor,
                                         decoded_image_tensor, resized_image_tensor,
                                         bottleneck_tensor, FLAGS.tfhub_module)

        # Create the operations we need to evaluate the accuracy of our new layer.
        evaluation_step, _ = add_evaluation_step(final_tensor, ground_truth_input)

        # Merge all the summaries and write them out to the summaries_dir
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                             sess.graph)

        validation_writer = tf.summary.FileWriter(
            FLAGS.summaries_dir + '/validation')

        # Create a train saver that is used to restore values into an eval graph
        # when exporting models.
        train_saver = tf.train.Saver()

        # Run the training for as many cycles as requested on the command line.
        for i in range(FLAGS.how_many_training_steps):
            # Get a batch of input bottleneck values, either calculated fresh every
            # time with distortions applied, or from the cache stored on disk.
            if do_distort_images:
                (train_bottlenecks,
                 train_ground_truth) = bottleneck.get_random_distorted_bottlenecks(
                    sess, image_lists, FLAGS.train_batch_size, 'training',
                    FLAGS.image_dir, distorted_jpeg_data_tensor,
                    distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
            else:
                (train_bottlenecks,
                 train_ground_truth, _) = bottleneck.get_random_cached_bottlenecks(
                    sess, image_lists, FLAGS.train_batch_size, 'training',
                    FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                    decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                    FLAGS.tfhub_module)
            # Feed the bottlenecks and ground truth into the graph, and run a training
            # step. Capture training summaries for TensorBoard with the `merged` op.
            train_summary, _ = sess.run(
                [merged, train_step],
                feed_dict={bottleneck_input: train_bottlenecks,
                           ground_truth_input: train_ground_truth})
            train_writer.add_summary(train_summary, i)

            # Every so often, print out how well the graph is training.
            is_last_step = (i + 1 == FLAGS.how_many_training_steps)
            if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
                train_accuracy, cross_entropy_value = sess.run(
                    [evaluation_step, cross_entropy],
                    feed_dict={bottleneck_input: train_bottlenecks,
                               ground_truth_input: train_ground_truth})
                logging.info('%s: Step %d: Train accuracy = %.1f%%',
                             datetime.now(), i, train_accuracy * 100)
                logging.info('%s: Step %d: Cross entropy = %f',
                             datetime.now(), i, cross_entropy_value)
                # TODO: Make this use an eval graph, to avoid quantization
                # moving averages being updated by the validation set, though in
                # practice this makes a negligable difference.
                validation_bottlenecks, validation_ground_truth, _ = (
                    bottleneck.get_random_cached_bottlenecks(
                        sess, image_lists, FLAGS.validation_batch_size, 'validation',
                        FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                        decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                        FLAGS.tfhub_module))
                # Run a validation step and capture training summaries for TensorBoard
                # with the `merged` op.
                validation_summary, validation_accuracy = sess.run(
                    [merged, evaluation_step],
                    feed_dict={bottleneck_input: validation_bottlenecks,
                               ground_truth_input: validation_ground_truth})
                validation_writer.add_summary(validation_summary, i)
                logging.info('%s: Step %d: Validation accuracy = %.1f%% (N=%d)',
                             datetime.now(), i, validation_accuracy * 100,
                             len(validation_bottlenecks))

            # Store intermediate results
            intermediate_frequency = FLAGS.intermediate_store_frequency

            if (intermediate_frequency > 0 and (i % intermediate_frequency == 0)
                    and i > 0):
                # If we want to do an intermediate save, save a checkpoint of the train
                # graph, to restore into the eval graph.
                train_saver.save(sess, FLAGS.checkpoint_path)
                intermediate_file_name = (FLAGS.intermediate_output_graphs_dir +
                                          'intermediate_' + str(i) + '.pb')
                logging.info('Save intermediate result to : %s', intermediate_file_name)
                save_graph_to_file(intermediate_file_name, module_spec,
                                   class_count)

        # After training is complete, force one last save of the train checkpoint.
        train_saver.save(sess, FLAGS.checkpoint_path)

        # We've completed all our training, so run a final test evaluation on
        # some new images we haven't used before.
        run_final_eval(sess, module_spec, class_count, image_lists,
                       jpeg_data_tensor, decoded_image_tensor, resized_image_tensor,
                       bottleneck_tensor)

        # Write out the trained graph and labels with the weights stored as
        # constants.
        logging.info('Save final result to : %s', FLAGS.output_graph)
        if wants_quantization:
            logging.info('The model is instrumented for quantization with TF-Lite')
        save_graph_to_file(FLAGS.output_graph, module_spec, class_count)
        with tf.gfile.GFile(FLAGS.output_labels, 'w') as f:
            f.write('\n'.join(image_lists.keys()) + '\n')

        if FLAGS.saved_model_dir:
            export_model(module_spec, class_count, FLAGS.saved_model_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        type=str,
        default='',
        help='Path to folders of labeled images.'
    )
    parser.add_argument(
        '--output_graph',
        type=str,
        default='/tmp/output_graph.pb',
        help='Where to save the trained graph.'
    )
    parser.add_argument(
        '--intermediate_output_graphs_dir',
        type=str,
        default='/tmp/intermediate_graph/',
        help='Where to save the intermediate graphs.'
    )
    parser.add_argument(
        '--intermediate_store_frequency',
        type=int,
        default=0,
        help="""\
         How many steps to store intermediate graph. If "0" then will not
         store.\
      """
    )
    parser.add_argument(
        '--output_labels',
        type=str,
        default='/tmp/output_labels.txt',
        help='Where to save the trained graph\'s labels.'
    )
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default='/tmp/retrain_logs',
        help='Where to save summary logs for TensorBoard.'
    )
    parser.add_argument(
        '--how_many_training_steps',
        type=int,
        default=4000,
        help='How many training steps to run before ending.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='How large a learning rate to use when training.'
    )
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=10,
        help='What percentage of images to use as a test set.'
    )
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help='What percentage of images to use as a validation set.'
    )
    parser.add_argument(
        '--eval_step_interval',
        type=int,
        default=10,
        help='How often to evaluate the training results.'
    )
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=100,
        help='How many images to train on at a time.'
    )
    parser.add_argument(
        '--test_batch_size',
        type=int,
        default=-1,
        help="""\
      How many images to test on. This test set is only used once, to evaluate
      the final accuracy of the model after training completes.
      A value of -1 causes the entire test set to be used, which leads to more
      stable results across runs.\
      """
    )
    parser.add_argument(
        '--validation_batch_size',
        type=int,
        default=100,
        help="""\
      How many images to use in an evaluation batch. This validation set is
      used much more often than the test set, and is an early indicator of how
      accurate the model is during training.
      A value of -1 causes the entire validation set to be used, which leads to
      more stable results across training iterations, but may be slower on large
      training sets.\
      """
    )
    parser.add_argument(
        '--print_misclassified_test_images',
        default=False,
        help="""\
      Whether to print out a list of all misclassified test images.\
      """,
        action='store_true'
    )
    parser.add_argument(
        '--bottleneck_dir',
        type=str,
        default='/tmp/bottleneck',
        help='Path to cache bottleneck layer values as files.'
    )
    parser.add_argument(
        '--final_tensor_name',
        type=str,
        default='final_result',
        help="""\
      The name of the output classification layer in the retrained graph.\
      """
    )
    parser.add_argument(
        '--flip_left_right',
        default=False,
        help="""\
      Whether to randomly flip half of the training images horizontally.\
      """,
        action='store_true'
    )
    parser.add_argument(
        '--random_crop',
        type=int,
        default=0,
        help="""\
      A percentage determining how much of a margin to randomly crop off the
      training images.\
      """
    )
    parser.add_argument(
        '--random_scale',
        type=int,
        default=0,
        help="""\
      A percentage determining how much to randomly scale up the size of the
      training images by.\
      """
    )
    parser.add_argument(
        '--random_brightness',
        type=int,
        default=0,
        help="""\
      A percentage determining how much to randomly multiply the training image
      input pixels up or down by.\
      """
    )
    parser.add_argument(
        '--tfhub_module',
        type=str,
        default=(
            'https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/2'),
        help="""\
      Which TensorFlow Hub module to use. For more options,
      search https://tfhub.dev for image feature vector modules.\
      """)
    parser.add_argument(
        '--saved_model_dir',
        type=str,
        default='',
        help='Where to save the exported graph.')
    parser.add_argument(
        '--logging_verbosity',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'],
        help='How much logging output should be produced.')
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='/tmp/_retrain_checkpoint',
        help='Where to save checkpoint files.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
