import urllib.request
import bot as robbot
from Image import Image
import time
import numpy


def pool(bot):
    sleep_count = 0
    while True:
        messages = bot.get_unanswered_messages(20)
        if messages["count"] > 0:
            # if we just got a message our bot wont sleep for a long time to get another one
            sleep = 1
            sleep_count = 0
            from_id = messages["items"][0]["last_message"]["from_id"]
            body = messages["items"][0]["last_message"]["text"]
            print(body)

            if 'msg:' in body.lower():
                msg = body[5:]
                robbot.Voice.say(msg)
                bot.send_text_message(from_id, "ok")
                time.sleep(sleep)
                continue

            if 'num:' in body.lower():
                try:
                    label = int(body[4:].strip(' '))
                    image.save_image(network, label)
                    bot.send_photo_message(from_id, [f'img_num_10/{label}.png'])
                    time.sleep(sleep)
                    continue
                except Exception as e:
                    raise e

            image_url = None
            try:
                image_url = messages["items"][0]["last_message"]["attachments"][0]['photo']['sizes'][0]['url']
                image_name = 'images/'+str(messages["items"][0]["last_message"]["attachments"][0]['photo']['id'])+'.png'
            except IndexError:
                bot.default_message(from_id)

            if image_url:
                urllib.request.urlretrieve(image_url, image_name)

                data = image.prepare_image(image_name)
                outputs = network.query(data)
                label = numpy.argmax(outputs)

                bot.send_text_message(from_id, f"Это цифра {label}")
        else:
            # if no one writes to us, then we can sleep a little longer
            sleep_count += 1
            sleep = 5 if sleep_count == 10 else 1
        time.sleep(sleep)


def main():
    global network
    global image
    network = robbot.neuralNetwork.load_network()
    image = Image()
    bot = robbot.Vk(robbot.config['vk']['token'], lang='ru')
    pool(bot)


if __name__ == '__main__':
    main()
