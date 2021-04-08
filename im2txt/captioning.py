import math
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
from collections import defaultdict

from im2txt.im2txt import configuration
from im2txt.im2txt import inference_wrapper
from im2txt.im2txt.inference_utils import caption_generator
from im2txt.im2txt.inference_utils import vocabulary

# TODO find out how to change parameters
# TODO evaluate how bad the mis-classification is


class Captioning:
    def __init__(self, model_checkpoint="./checkpoints/5M_iterations/model.ckpt-5000000",
                 word_counts="./vocab/word_counts.txt", beam_size=3, max_caption_length=40, length_normalization_factor=0.0):
        g = tf.Graph()
        with g.as_default():
            model = inference_wrapper.InferenceWrapper()
            restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                        model_checkpoint)
        g.finalize()
        # Create the vocabulary.
        self.vocab = vocabulary.Vocabulary(word_counts)
        
        self.sess = tf.Session(graph=g)

        # Load the model from checkpoint.
        restore_fn(self.sess)
        self.generator = caption_generator.CaptionGenerator(model, self.vocab, beam_size=beam_size,
                                                            max_caption_length=max_caption_length, length_normalization_factor=length_normalization_factor)

    def image(self, filename):
        """

        :param filename:
        :return:
        """
        # TODO finalize doc
        result_set = defaultdict(list)
        count = 0
        with tf.gfile.GFile(filename, "rb") as f:
            image = f.read()
        captions = self.generator.beam_search(self.sess, image)
        # print("Captions for image %s:" % os.path.basename(filename))
        for i, caption in enumerate(captions):
            # Ignore begin and end words.
            prob = math.exp(caption.logprob)
            if prob > 0:
                count +=1
                sentence = [self.vocab.id_to_word(w) for w in caption.sentence[1:-1]]
                sentence = " ".join(sentence)
                result_set[str(count)]={"Sentence":sentence,"confidence":prob}
                # print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
        return result_set
                
    def close_session(self):
        self.sess.close()
