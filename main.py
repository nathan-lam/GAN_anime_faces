from GAN import *
from data_prep import *
import os
import time

attempt = 4
EPOCHS = 300
# noise_dim is set in GAN.py
num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate, noise_dim])

generator = make_generator_model()
discriminator = make_discriminator_model()

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = f'Attempt{attempt}/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

def trunc(x):  # truncates decimal
    return float('%.6f' % x)


def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])  # generate noise

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:  # idk why
        generated_images = generator(noise, training=True)  # feed noise into generator

        real_output = discriminator(images, training=True)  # feed real into discriminator
        fake_output = discriminator(generated_images, training=True)  # feed generated into discriminator

        gen_loss = generator_loss(fake_output)  # get generator loss
        disc_loss = discriminator_loss(real_output, fake_output)  # get discriminator loss

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return np.array([[gen_loss,
                      disc_loss,
                      real_output.numpy()[0, 0],
                      fake_output.numpy()[0, 0]
                      ]], dtype=float)


def train(dataset, epochs):
    total_epoch_stats = {"Generator_Loss": [],  # Generator Loss
                         "Discriminator_Loss": [],  # Discriminator Loss
                         "Discriminating_Real": [],  # average decision probability given a real
                         "Discriminating_Fake": [],
                         "Times": []}  # average decision probability given a fake
    n = BATCH_SIZE

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        start = time.time()
        epoch_stats = np.zeros((n, 4), dtype=float)

        i = 0
        for image_batch in dataset:
            # np.concatenate((epoch_stats, train_step(image_batch)), axis=0)
            epoch_stats[i, :] = train_step(image_batch)
            i += 1
        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed)

        # Save the model every 10 epochs
        if (epoch + 1) % (epochs // 10) == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        epoch_time = time.time() - start
        mean_epoch_stats = np.mean(epoch_stats, axis=0)

        print(
            f'\t Time = {trunc(epoch_time)} sec - Gen Loss = {trunc(mean_epoch_stats[0])} - Disc Loss = {trunc(mean_epoch_stats[1])}')
        print(
            f"\t D(real|given real) = {trunc(mean_epoch_stats[2])} - D(fake|given fake) = {trunc(mean_epoch_stats[3])}")

        stat_names = list(total_epoch_stats.keys())
        for j in range(len(stat_names) - 1):
            total_epoch_stats[stat_names[j]].append(mean_epoch_stats[j])
        total_epoch_stats["Times"].append(epoch_time)

    # checkpointing the very last step
    checkpoint.save(file_prefix=checkpoint_prefix)
    generate_and_save_images(generator,
                             epochs,
                             seed)

    return total_epoch_stats


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    size = test_input.shape[0] ** 0.5
    assert size % 1 == 0, "Please generate a square number of examples"
    size = int(size)
    fig = plt.figure(figsize=(size, size))

    for i in range(predictions.shape[0]):
        plt.subplot(size, size, i + 1)
        plt.imshow(tf.cast(predictions[i] * 127.5 + 127.5, tf.int32))
        plt.axis('off')

    plt.savefig('Attempt{}/checkpoint_images/image_at_epoch_{:04d}.png'.format(attempt,epoch))
    plt.close('all')


def plot_graphs(history, metric):
    plt.plot(history[metric])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric])
    plt.ylim(None, np.max(history[metric]) * 1.001)


history = train(train_dataset, EPOCHS)
plt.close('all')

# plotting stats per epoch
for stat in history:
    plot_graphs(history, stat)
    plt.savefig(f"Attempt{attempt}/{stat}_per_epoch.png")
    plt.close('all')

generate_and_save_images(generator, 9999, tf.random.normal([num_examples_to_generate, noise_dim]))

