import vk_api
import os
import numpy
import pickle
import scipy.special
import scipy.ndimage
import json


with open('config.json') as f:
    config = json.load(f)


class Voice:

    @staticmethod
    def say(message):
        os.system(f'say {message}')


class Vk:

    def __init__(self, token, lang='eng'):
        self.token = token
        self.vk = vk_api.VkApi(token=token)
        self.vk_upload = vk_api.VkUpload(vk=self.vk)
        self.vk._auth_token()
        self.lang = lang
        if lang == 'eng':
            self.default_text_message = config['msg']['eng']['default']
        elif lang == 'ru':
            self.default_text_message = config['msg']['ru']['default']

    def default_message(self, from_id):
        self.send_text_message(from_id, self.default_text_message)

    def send_text_message(self, from_id, text):
        self.vk.method("messages.send", {"peer_id": from_id, "message": text})

    def send_photo_message(self, from_id, photos, send_text=True):
        res = self.vk_upload.photo_messages(photos)
        photos_id = []
        owner_id = res[0]['owner_id']
        for photo in res:
            photos_id.append(photo['id'])
        for photo_id in photos_id:
            self.vk.method("messages.send", {
                "peer_id": from_id,
                "attachment": f"photo{owner_id}_{photo_id}"
            })

        if send_text:
            self.send_text_message(from_id, 'My brain see it like this' if self.lang=='eng' else 'Мой мозг видит это так')

    def get_unanswered_messages(self, count=20):
        return self.vk.method("messages.getConversations", {"offset": 0, "count": count, "filter": "unanswered"})


class Telegram:
    pass


class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        self.lr = learningrate
        self.activation_function = lambda x: scipy.special.expit(x)

    @staticmethod
    def load_network(filename='network.pickle'):
        with open(filename, 'rb') as file:
            network = pickle.load(file)
        return network

    @staticmethod
    def inverse_activation_function(x):
        return scipy.special.logit(x)

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    # backquery the neural network
    # we'll use the same termnimology to each item,
    # eg target are the values at the right of the network, albeit used as input
    # eg hidden_output is the signal to the right of the middle nodes
    def backquery(self, targets_list):
        # transpose the targets list to a vertical array
        final_outputs = numpy.array(targets_list, ndmin=2).T

        # calculate the signal into the final output layer
        final_inputs = self.inverse_activation_function(final_outputs)

        # calculate the signal out of the hidden layer
        hidden_outputs = numpy.dot(self.who.T, final_inputs)
        # scale them back to 0.01 to .99
        hidden_outputs -= numpy.min(hidden_outputs)
        hidden_outputs /= numpy.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01

        # calculate the signal into the hidden layer
        hidden_inputs = self.inverse_activation_function(hidden_outputs)

        # calculate the signal out of the input layer
        inputs = numpy.dot(self.wih.T, hidden_inputs)
        # scale them back to 0.01 to .99
        inputs -= numpy.min(inputs)
        inputs /= numpy.max(inputs)
        inputs *= 0.98
        inputs += 0.01

        return inputs
