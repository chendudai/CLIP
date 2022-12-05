import numpy as np
import torch,os
from pkg_resources import packaging
import clip
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from PIL import Image
import fnmatch
import os
import json
from pathlib import Path
# import ipyplot
import matplotlib.pyplot as plt
import cv2


# Use Wikiscens Data
def Extract_WikiScenes_Data(path):

    # path = '/mnt/data/chendudai/repos/wikiscenes/data/WikiScenes/cathedrals/0'
    captions = []
    path_images = []
    cnt = 0
    long_captions = []
    list_categories = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*.json'):
            path_json = os.path.join(root, filename)

            with open(path_json, 'r', encoding='utf-8') as f:
                my_data = json.load(f)
            for image in my_data['pictures']:
                caption = my_data['pictures'][image]['caption']
                file_path = os.path.join(path_json[:-13],'pictures', image)

                try:
                    clip.tokenize(caption)
                except:
                    cnt = cnt + 1
                    long_captions.append(caption)
                    continue


                if not isEnglish(caption) or caption == '' or not os.path.isfile(file_path): # or len(caption) > 77
                    continue

                captions.append(caption)
                path_images.append(file_path)

                # Extract Categories For The Images
                categories = []
                path_categories = Path(file_path)
                while path_categories.name != 'cathedrals':
                    son_folder = path_categories.name
                    path_categories = path_categories.parent
                    if path_categories.name == 'pictures':
                        path_categories = path_categories.parent
                        continue

                    for filename_category in fnmatch.filter(filenames, '*.json'):
                        path_json_category = os.path.join(path_categories, filename_category)
                        with open(path_json_category, 'r', encoding='utf-8') as f:
                            folder_data = json.load(f)

                        category = list(folder_data['pairs'].keys())[int(son_folder)]
                        categories.append(category)

                list_categories.append(categories)

    print("len[captions] = ",len(captions))

    # with open('/mnt/data/chendudai/repos/CLIP/save/captions.txt', 'w') as fp:
    #     for item in captions:
    #         fp.write("%s\n" % item)
    #
    # with open('/mnt/data/chendudai/repos/CLIP/save/path_images.txt', 'w') as fp:
    #     for item in path_images:
    #         fp.write("%s\n" % item)
    #
    # with open('/mnt/data/chendudai/repos/CLIP/save/long_captions.txt', 'w') as fp:
    #     for item in long_captions:
    #         fp.write("%s\n" % item)
    #
    # with open('/mnt/data/chendudai/repos/CLIP/save/list_categories.txt', 'w') as fp:
    #     for item in list_categories:
    #         fp.write("%s\n" % item)

    return captions, list_categories, path_images, long_captions



# -*- coding: utf-8 -*-
def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def saveBestImagesForCaption(probs_texts, path_images, captions, text_index, categories, use_finetuned_model, mode_to_check):
    argsorted_images = probs_texts[0, :].argsort()
    best_images = argsorted_images[-11:]
    best_images = best_images[::-1]  # Flip
    images_list = [path_images[i] for i in best_images]
    images_captions = [captions[i] for i in best_images]


    # Plot
    columns = 3
    rows = 4
    fig = plt.figure(figsize=(8, 8))
    for i in range(1, columns * rows + 1):

        fig.add_subplot(rows, columns, i)
        if i == 1:
            img = cv2.imread(path_images[text_index])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(rows, columns, i).set_title('real image', size=10)
        else:
            img = cv2.imread(images_list[i - 2])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(rows, columns, i).set_title(f'{i - 1}_' + images_captions[i - 2], wrap=True, size=5)

        fig.axes[i - 1].get_xaxis().set_visible(False)
        fig.axes[i - 1].get_yaxis().set_visible(False)

        # plt.tight_layout()
        plt.imshow(img)


    fig.suptitle('Caption: ' + captions[text_index], wrap=True)
    # plt.tight_layout()

    if use_finetuned_model:
        path_to_save = './save/figures/captions_withFineTune/' + '/'.join(categories[text_index][::-1])
    else:
        path_to_save = './save/figures/captions_withoutFineTune/' + '/'.join(categories[text_index][::-1])

    path_to_save = path_to_save.replace('\'', '')
    path_to_save = path_to_save.replace(')', '')
    path_to_save = path_to_save.replace('(', '')
    path_to_save = path_to_save.replace('-', '')
    path_to_save = path_to_save.replace('  ', '_')
    path_to_save = path_to_save.replace(' ', '_')

    os.makedirs(path_to_save, exist_ok=True)

    if mode_to_check == 0:
        plt.savefig(path_to_save + '/' + str(text_index) + '.png')
    elif mode_to_check == 1:
        plt.savefig(path_to_save + '/' + 'categories_full_' + str(text_index) + '.png')
    elif mode_to_check == 2:
        plt.savefig(path_to_save + '/' + 'categories_last_' + str(text_index) + '.png')

    plt.close(fig)

def calcClipRes(probs_texts, path_images, captions, text_index):
    argsorted_images = probs_texts[0, :].argsort()
    best_images = argsorted_images[-11:]
    best_images = best_images[::-1]  # Flip
    images_list = [path_images[i] for i in best_images]
    images_captions = [captions[i] for i in best_images]

    res_top1, res_top5, res_top10 = 0, 0, 0
    if text_index in best_images[0:10]:
        res_top10 = 1
    if text_index in best_images[0:5]:
        res_top5 = 1
    if text_index == best_images[0]:
        res_top1 = 1

    return res_top1, res_top5, res_top10

def get_probs(model, images, texts):
    logits_per_image, logits_per_text = model(images, texts)
    probs_images = logits_per_image.softmax(dim=-1).cpu().detach().numpy()
    probs_texts = logits_per_text.softmax(dim=-1).cpu().detach().numpy()
    return probs_images, probs_texts

# https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


#Define Function
class image_title_dataset():
    def __init__(self, list_image_path,list_txt, preprocess):
        self.list_txt = list_txt
        self.image_path = list_image_path
        self.title  = clip.tokenize(list_txt)
        self.preprocess = preprocess
        #you can tokenize everything at once in here(slow at the beginning), or tokenize it in the training loop.

    def __len__(self):
        return len(self.list_txt)

    def __getitem__(self, idx):
        image = self.preprocess(Image.open(self.image_path[idx])) # Image from PIL module
        title = self.title[idx]
        return image,title




def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
    print(device)
    use_finetuned_model = True
    mode_to_check = 0  # 0 is captions, 1 is categories_full, 2 is categoreis_last, 3 is unique_captions, 4 is unique_categories

    save_dir = './save'
    dataset_dir = './dataset/cathedrals/0'

    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)  # Must set jit=False for training

    if use_finetuned_model:
        checkpoint = torch.load('./save/lr5e-7_clip_finetuned_60Epcohs.pt')
        model.load_state_dict(checkpoint['model_state_dict'])


    BATCH_SIZE = 64 # BATCH_SIZE must larger than 1
    EPOCH = 100

    classes = ['facade', 'window', 'chapel', 'organ', 'nave', 'tower', 'choir', 'portal', 'altar', 'statue']

    captions, list_categories, path_images, long_captions = \
        Extract_WikiScenes_Data(dataset_dir)

    if device == "cpu":
        model.float()
    else:
        clip.model.convert_weights(model)  # Actually this line is unnecessary since clip by default already on float16


    # Create unique list of captions and contains classes
    unique_captions = []
    unique_path_images = []
    unique_categories = []
    for i, caption in enumerate(captions):
        if caption not in unique_captions and any(substring.lower() in caption.lower() for substring in classes):
            unique_captions.append(caption)
            unique_path_images.append(path_images[i])
            unique_categories.append(', '.join(list_categories[i]))


    # Split train and test captions
    test_captions = []
    train_captions = []
    test_path_images = []
    train_path_images = []
    train_categories_full = []
    train_categories_last = []
    test_categories_full = []
    test_categories_last = []
    list_test_categories = []
    list_train_categories = []

    for i,caption in enumerate(captions):
        if i % 10 == 0:
            test_captions.append(caption)
            test_path_images.append(path_images[i])
            list_test_categories.append(list_categories[i])
            test_categories_last.append(list_categories[i][0])
            test_categories_full.append(', '.join(list_categories[i]))
        else:
            train_captions.append(caption)
            train_path_images.append(path_images[i])
            list_train_categories.append(list_categories[i])
            train_categories_last.append(list_categories[i][0])
            train_categories_full.append(', '.join(list_categories[i]))


    # change text and images path according to the mode
    if mode_to_check == 0:
        text_to_use = test_captions
    elif mode_to_check == 1:
        text_to_use = test_categories_full
    elif mode_to_check == 2:
        text_to_use = test_categories_last
    elif mode_to_check == 3:
        text_to_use = unique_captions
        test_path_images = unique_path_images
    elif mode_to_check == 4:
        text_to_use = unique_categories
        test_path_images = unique_path_images


    # Prepare texts and images to get_probs
    images_4D = []
    for image in test_path_images:
        x = preprocess(Image.open(image)).unsqueeze(0).to(device)
        images_4D.append(x)
    images_4D = torch.cat(images_4D, dim=0)


    texts = clip.tokenize(text_to_use).to(device)
    res_top1_ctr, res_top5_ctr, res_top10_ctr = 0, 0, 0
    for text_index in range(len(text_to_use)):
        print(text_index)
        probs_images, probs_texts = get_probs(model, images_4D, texts[text_index].unsqueeze(dim=0))
        try:
            saveBestImagesForCaption(probs_texts, test_path_images, text_to_use, text_index, list_test_categories, use_finetuned_model, mode_to_check)
            res_top1, res_top5, res_top10 = calcClipRes(probs_texts, test_path_images, text_to_use, text_index)
            res_top1_ctr += res_top1
            res_top5_ctr += res_top5
            res_top10_ctr += res_top10

        except Exception as e:
            print(e)
            continue


    print(res_top1_ctr)
    print(res_top5_ctr)
    print(res_top10_ctr)



    # Prepare dataset for training
    dataset = image_title_dataset(train_path_images, train_captions, preprocess)
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)  # Define your own dataloader

    # Train - Fine Tuning
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-7, betas=(0.9, 0.98), eps=1e-6,
                           weight_decay=0.2)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

    for epoch in range(EPOCH):
        for batch in train_dataloader:
            optimizer.zero_grad()

            images, texts = batch

            images = images.to(device)
            texts = texts.to(device)

            logits_per_image, logits_per_text = model(images, texts)

            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

            total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            total_loss.backward()
            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

            print("total_loss = ", total_loss)

        print("[", epoch, "]\t total_loss = ", total_loss)

        if epoch % 10 == 0:

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
            }, os.path.join(save_dir, "lr5e-7_clip_finetuned_" + str(epoch) + "Epcohs.pt"))


if __name__ == "__main__":
    main()