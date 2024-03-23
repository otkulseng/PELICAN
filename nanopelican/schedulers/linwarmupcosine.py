from keras.optimizers.schedules import CosineDecayRestarts, LearningRateSchedule, ExponentialDecay
import tensorflow as tf

@tf.keras.saving.register_keras_serializable(name='CosineAnnealingExpDecay')
class LinearWarmupCosineAnnealing(LearningRateSchedule):
    """Schedule like the one proposed in https://arxiv.org/pdf/2310.16121.pdf

    Args:
        LearningRateSchedule (_type_): _description_
    """
    def __init__(self, epochs, steps_per_epoch) -> None:
        super().__init__()

        assert(epochs > 4 + 12) # Does not really make sense without sufficient epochs

        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs

        self.warmup_target = 0.001
        self.cosine_decay = CosineDecayRestarts(
            initial_learning_rate=self.warmup_target,
            first_decay_steps=4*steps_per_epoch,
            t_mul=2,
        )

        self.exp_decay = ExponentialDecay(
            initial_learning_rate=self.warmup_target,
            decay_steps=steps_per_epoch,
            decay_rate=0.5
        )

    def __call__(self, step):
        epoch = step // self.steps_per_epoch

        def warmup():
            fraction = step / (4 * self.steps_per_epoch)
            return tf.cast(self.warmup_target * fraction, dtype=tf.float32)
        def cosine():
            return self.cosine_decay(step - 4 * self.steps_per_epoch)
        def expdecay():
            return self.exp_decay(step - (self.epochs - 12) * self.steps_per_epoch)


        return tf.case(
            [(tf.less(epoch, 4), warmup), (tf.less(epoch, self.epochs-12), cosine)],
            default=expdecay
        )

    def get_config(self):
        config = {
                'epochs': self.epochs,
                'steps_per_epoch': self.steps_per_epoch,
        }
        return config



