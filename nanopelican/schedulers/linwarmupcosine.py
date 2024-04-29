from keras.optimizers.schedules import CosineDecayRestarts, LearningRateSchedule, ExponentialDecay
import tensorflow as tf
import keras

@keras.saving.register_keras_serializable(name='CosineAnnealingExpDecay')
class LinearWarmupCosineAnnealing(LearningRateSchedule):
    """Schedule like the one proposed in https://arxiv.org/pdf/2310.16121.pdf

    Args:
        LearningRateSchedule (_type_): _description_
    """
    def __init__(self, epochs, steps_per_epoch) -> None:
        super().__init__()

        self.warmup_epochs = 4
        self.exp_decay_epochs = 12
        # assert(epochs > self.warmup_epochs + self.exp_decay_epochs) # Does not really make sense without sufficient epochs

        if type(steps_per_epoch) == dict:
            print(steps_per_epoch)
            steps_per_epoch = steps_per_epoch['config']['value']

        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs


        self.init_lr = 0.001
        self.warmup_target = 0.01
        self.min_lr = 1e-6
        self.cosine_decay = CosineDecayRestarts(
            initial_learning_rate=self.warmup_target,
            first_decay_steps=self.warmup_epochs*steps_per_epoch,
            t_mul=2,
            alpha=self.min_lr/self.warmup_target,
        )

        self.exp_decay = ExponentialDecay(
            initial_learning_rate=self.min_lr,
            decay_steps=steps_per_epoch,
            decay_rate=0.5
        )

    def __call__(self, step):
        epoch = step // self.steps_per_epoch

        def warmup():
            fraction = step / (self.warmup_epochs * self.steps_per_epoch)
            return tf.cast(self.init_lr +(self.warmup_target - self.init_lr) * fraction, dtype=tf.float32)
        def cosine():
            return self.cosine_decay(step - 4 * self.steps_per_epoch)
        def expdecay():
            return self.exp_decay(step - (self.epochs - self.exp_decay_epochs) * self.steps_per_epoch)


        return tf.case(
            [(tf.less(epoch, self.warmup_epochs), warmup), (tf.less(epoch, self.epochs-self.exp_decay_epochs), cosine)],
            default=expdecay
        )

    def get_config(self):
        config = {
                'epochs': self.epochs,
                'steps_per_epoch': self.steps_per_epoch,
        }
        return config



