import os
import torch
import torch.nn as nn
import json
from torch.autograd import Variable
from tqdm import tqdm
import h5py
import config
import argparse
import eval
from data import VQA
import model
import utils
import data
from resnet import resnet as caffe_resnet
import torch.utils.data as torch_data

def create_coco_loader(*paths):
    transform = utils.get_transform(config.image_size, config.central_fraction)
    datasets = [data.CocoImages(path, transform=transform) for path in paths]
    dataset = data.Composite(*datasets)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.preprocess_batch_size,
        num_workers=config.data_workers,
        shuffle=False,
        pin_memory=True,
    )
    return data_loader

class VAC(VQA):
    def __init__(self):
        with open(config.vocabulary_path, 'r') as fd:
            vocab_json = json.load(fd)

        # vocab
        self.vocab = vocab_json
        self.token_to_index = self.vocab['question']

        with open(config.vocabularyi_path, 'r') as fd:
            vocabi_json = json.load(fd)
        self.index_to_answer = vocabi_json['answeri']
        #print("index_to_answer:", self.index_to_answer)
        #print("index_to_answer 2999:", self.index_to_answer['{}'.format(2999)])

    def _decode_answer(self, index):
        return self.index_to_answer.get('{}'.format(index))#['2394']


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = caffe_resnet.resnet152(pretrained=True)

        def save_output(module, input, output):
            self.buffer = output
        self.model.layer4.register_forward_hook(save_output)

    def forward(self, x):
        self.model(x)
        return self.buffer

def main(args):
    log = torch.load(args.model_file)
    tokens = len(log['vocab']['question']) + 1
    net = torch.nn.DataParallel(model.Net(tokens)).cuda()
    net.load_state_dict(log['weights'])

    rnet = Net().cuda()
    rnet.eval()

    loader = create_coco_loader(args.image_path)
    features_shape = (
        len(loader.dataset),
        config.output_features,
        config.output_size,
        config.output_size
    )

    if os.path.isfile(config.preprocessed_path):
        os.remove(config.preprocessed_path)

    with h5py.File(config.preprocessed_path, libver='latest') as fd:
        features = fd.create_dataset('features', shape=features_shape, dtype='float16')

        for ids, imgs in tqdm(loader):
            imgs = Variable(imgs.cuda(async=True), volatile=True)
            out = rnet(imgs)
            feature = out.data.cpu().numpy().astype('float16')
            img = feature[0].astype('float32')
            img = [torch.from_numpy(img)]
            v = torch_data.dataloader.default_collate(img)
            print("v: ", v)

    #num_tokens = 15193 #this is from coco caption
    #net = nn.DataParallel(model.Net(num_tokens)).cuda()
    vac = VAC()
    max_question_length = 23
    vac._max_length = max_question_length
    question = input("Please input your problem: ")
    #print(len(question))
    assert(len(question)<vac._max_length)
    question = question.lower()[:-1]
    question = question.split(' ') #convet a string to a list like ['what', 'is', 'the', 'table', 'made', 'of']
    #q, q_length  = vac._encode_question(question)
    questions = [vac._encode_question(question)]
    qs = torch_data.dataloader.default_collate(questions)

    var_params = {
        'volatile': True,
        'requires_grad': False,
    }

    print("qs:", qs)
    q, q_len = qs
    out = net(v, q, q_len)
    print("out:",out)
    answ = []
    _, answer = out.data.cpu().max(dim=1)
    print("answer shape: ",answer.shape)
    answ.append(answer.view(-1))
    if (len(answ) > 0):
        answ = list(torch.cat(answ, dim=0))
    # Print out the image and the generated caption
    print("answer:", answ[0].data)
    print("answer type:", answ[0].type())
    answtxt = vac._decode_answer(answ[0].item())
    print("answtxt:", answtxt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='input image for generating caption')
    parser.add_argument('--model_file', type=str, default='logs/2017-08-04_00.55.19.pth',
                        help='path for model file')
    args = parser.parse_args()
    main(args)