import os
import pickle
from collections import defaultdict

import copy
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from .utils import download_detection
from torch.utils.data import Dataset

RAW_DATA_FOLDER = 'faces_aligned_small_mirrored_co_aligned_cropped_cleaned'
RESOLUTION = 32
ID_HELD_OUT = 0.1
OOD_PROPORTION = 1.0



def preprocess_reduced_train_set(args):
    print(f'Preprocessing reduced train proportion dataset and saving to yearbook_{args.reduced_train_prop}.pkl')
    np.random.seed(0)

    orig_data_file = os.path.join(args.data_dir, f'yearbook.pkl')
    dataset = pickle.load(open(orig_data_file, 'rb'))
    years = list(sorted(dataset.keys()))
    train_fraction = args.reduced_train_prop / (1 - ID_HELD_OUT)

    for year in years:
        train_images = dataset[year][0]['images']
        train_labels = dataset[year][0]['labels']

        num_train_samples = len(train_labels)
        reduced_num_train_samples = int(train_fraction * num_train_samples)
        idxs = np.random.permutation(np.arange(num_train_samples))
        train_idxs = idxs[:reduced_num_train_samples].astype(int)

        new_train_images = np.array(train_images)[train_idxs]
        new_train_labels = np.array(train_labels)[train_idxs]
        dataset[year][0]['images'] = np.stack(new_train_images, axis=0) / 255.0
        dataset[year][0]['labels'] = np.array(new_train_labels)

    preprocessed_data_file = os.path.join(args.data_dir, f'yearbook_{args.reduced_train_prop}.pkl')
    pickle.dump(dataset, open(preprocessed_data_file, 'wb'))
    np.random.seed(args.random_seed)


def preprocess_orig(args):
    print(f'Preprocessing dataset and saving to yearbook.pkl')
    np.random.seed(0)
    raw_data_path = os.path.join(args.data_dir, RAW_DATA_FOLDER)
    if not os.path.exists(raw_data_path):
        raise ValueError(f'{RAW_DATA_FOLDER} is not in the data directory {args.data_dir}!')

    path = os.path.join(args.data_dir, RAW_DATA_FOLDER)
    dir_M = os.listdir(f'{path}/M')
    print('num male photos', len(dir_M))
    dir_F = os.listdir(f'{path}/F')
    print('num female photos', len(dir_F))

    images = defaultdict(list)
    labels = defaultdict(list)
    year_counts = {}
    for item in dir_M:
        year = int(item.split('_')[0])
        img = f'{path}/M/{item}'
        if os.path.isfile(img):
            img = Image.open(img)
            img_resize = img.resize((RESOLUTION, RESOLUTION), Image.ANTIALIAS)
            img_resize.save(f'{args.data_dir}/yearbook/{item}', 'PNG')
            images[year].append(np.array(img_resize))
            labels[year].append(0)
            if year not in year_counts.keys():
                year_counts[year] = {}
                year_counts[year]['m'] = 0
                year_counts[year]['f'] = 0
            year_counts[year]['m'] += 1

    for item in dir_F:
        year = int(item.split('_')[0])
        img = f'{path}/F/{item}'
        if os.path.isfile(img):
            img = Image.open(img)
            img_resize = img.resize((RESOLUTION, RESOLUTION), Image.ANTIALIAS)
            img_resize.save(f'{args.data_dir}/yearbook/{item}', 'PNG')
            images[year].append(np.array(img_resize))
            labels[year].append(1)
            if year not in year_counts.keys():
                year_counts[year] = {}
                year_counts[year]['m'] = 0
                year_counts[year]['f'] = 0
            year_counts[year]['f'] += 1

    dataset = {}
    for year in sorted(list(year_counts.keys())):
        # Ignore years 1905 - 1929, in yearbook.pkl: start from 1930 to 2013; for each year, use 0.9/0.1
        if year < 1930:
            del year_counts[year]
            continue
        dataset[year] = {}
        num_samples = len(labels[year])
        num_train_images = int((1 - ID_HELD_OUT) * num_samples)
        idxs = np.random.permutation(np.arange(num_samples))
        train_idxs = idxs[:num_train_images].astype(int)
        test_idxs = idxs[num_train_images:].astype(int)
        train_images = np.array(images[year])[train_idxs]
        train_labels = np.array(labels[year])[train_idxs]
        test_images = np.array(images[year])[test_idxs]
        test_labels = np.array(labels[year])[test_idxs]
        dataset[year][0] = {}
        dataset[year][0]['images'] = np.stack(train_images, axis=0) / 255.0
        dataset[year][0]['labels'] = np.array(train_labels)
        dataset[year][1] = {}
        dataset[year][1]['images'] = np.stack(test_images, axis=0) / 255.0
        dataset[year][1]['labels'] = np.array(test_labels)
        dataset[year][2] = {}
        dataset[year][2]['images'] = np.stack(images[year], axis=0) / 255.0
        dataset[year][2]['labels'] = np.array(labels[year])
       

    preprocessed_data_path = os.path.join(args.data_dir, 'yearbook.pkl')
    pickle.dump(dataset, open(preprocessed_data_path, 'wb'))
    np.random.seed(args.random_seed)

def preprocess(args):
    np.random.seed(0)
    if not os.path.isfile(os.path.join(args.data_dir, 'yearbook.pkl')):
        preprocess_orig(args)
    if args.reduced_train_prop is not None:
        if not os.path.isfile(os.path.join(args.data_dir, f'yearbook_{args.reduced_train_prop}.pkl')):
            preprocess_reduced_train_set(args)
    np.random.seed(args.random_seed)


class YearbookBase(Dataset):
    def __init__(self, args):
        super().__init__()

        if args.reduced_train_prop is None:
            self.data_file = f'{str(self)}.pkl'
        else:
            self.data_file = f'{str(self)}_{args.reduced_train_prop}.pkl'

        download_detection(args.data_dir, self.data_file)
        preprocess(args)

        self.datasets = pickle.load(open(os.path.join(args.data_dir, self.data_file), 'rb'))

        self.args = args
        self.num_classes = 2
        self.current_time = 0
        self.resolution = 32
        self.mini_batch_size = args.mini_batch_size
        self.mode = 0
        self.ssl_training = False

        self.ENV = list(sorted(self.datasets.keys())) #1930-2013
        self.num_tasks = len(self.ENV) #84
        self.num_examples = {i: len(self.datasets[i][self.mode]['labels']) for i in self.ENV} #for each year, number of train/test/all examples

        ## create a datasets object
        self.class_id_list = {i: {} for i in range(self.num_classes)}
        self.input_dim = []
        cumulative_batch_size = 0
        self.task_idxs = {}
        start_idx = 0

        np.random.seed(30)

        for year in self.ENV:  
            print("================================================================================================================")
            self.datasets[year][6] = {}
            self.datasets[year][6]['images'] = copy.deepcopy(self.datasets[year][2]['images'])  # 0.5 of all is assigned to test set, 0.5 of all is assigned to train+valid set
            self.datasets[year][6]['labels'] = copy.deepcopy(self.datasets[year][2]['labels'])
        
            images = self.datasets[year][2]['images']
            labels = self.datasets[year][2]['labels']
        
            num_samples = len(labels)
            num_train_valid_images = int(0.5 * num_samples)  # 50% for train+valid
            num_test_images = num_samples - num_train_valid_images  # Remaining 50% for test
            idxs = np.random.permutation(np.arange(num_samples))
            train_valid_idxs = idxs[:num_train_valid_images].astype(int)
            test_idxs_another = train_valid_idxs 
            test_idxs = idxs[num_train_valid_images:].astype(int)
            num_valid_images = int(0.1 * num_train_valid_images)  
        
            print("year", year)
            print("num_train_valid_images", num_train_valid_images)
            print("num_train", num_train_valid_images - num_valid_images)
            print("num_valid", num_valid_images)
            print("num_samples", num_samples)
        
            valid_idxs = train_valid_idxs[:num_valid_images]
            train_idxs = train_valid_idxs[num_valid_images:]
            train_images = np.array(images)[train_idxs]
            train_labels = np.array(labels)[train_idxs]
            valid_images = np.array(images)[valid_idxs]
            valid_labels = np.array(labels)[valid_idxs]
            test_images = np.array(images)[test_idxs]
            test_labels = np.array(labels)[test_idxs]
            test_images_another = np.array(images)[test_idxs_another]
            test_labels_another = np.array(labels)[test_idxs_another]
        
            subset_size = int(len(train_images) * OOD_PROPORTION)
            incremental_train_images = train_images[:subset_size]
            incremental_train_labels = train_labels[:subset_size]
            
            if year < 1970:
                self.datasets[year][3] = {}
                self.datasets[year][3]['images'] = self.datasets[year][0]['images']
                self.datasets[year][3]['labels'] = self.datasets[year][0]['labels']
                self.datasets[year][4] = {}
                self.datasets[year][4]['images'] = self.datasets[year][1]['images']
                self.datasets[year][4]['labels'] = self.datasets[year][1]['labels']
        
                print(f"before 1970, mode = 3, this year {year} train number is", len(self.datasets[year][3]['images']))
                print(f"before 1970, mode = 4, this year {year} valid number is", len(self.datasets[year][4]['images']))
        
            else:
                self.datasets[year][3] = {}
                self.datasets[year][3]['images'] = np.stack(incremental_train_images, axis=0) / 255.0
                self.datasets[year][3]['labels'] = np.array(incremental_train_labels)
                print(f"after 1970, mode = 3, this year {year} train number is", len(self.datasets[year][3]['labels']))
                print("this year incremental train_idxs is", train_idxs[:subset_size])
                print("show mode=3 labels")
                print(self.datasets[year][3]['labels'])
                
                self.datasets[year][5] = {}
                self.datasets[year][5]['images'] = np.stack(test_images, axis=0) / 255.0
                self.datasets[year][5]['labels'] = np.array(test_labels)
                print(f"after 1970, mode = 5, this year {year} test number is", len(self.datasets[year][5]['labels']))
                print("this year test_idxs is", test_idxs)


                self.datasets[year][7] = {}
                self.datasets[year][7]['images'] = np.stack(test_images_another, axis=0) / 255.0
                self.datasets[year][7]['labels'] = np.array(test_labels_another)
                print(f"after 1970, mode = 7, this year {year} another test number is", len(self.datasets[year][7]['labels']))
                print("this year another test_idxs is", test_idxs_another)
        
                self.datasets[year][4] = {}
                self.datasets[year][4]['images'] = np.stack(valid_images, axis=0) / 255.0
                self.datasets[year][4]['labels'] = np.array(valid_labels)
                print(f"after 1970, mode = 4, this year {year} valid number is", len(self.datasets[year][4]['labels']))
                print("this year valid_idxs is", valid_idxs)

        #print(self.datasets.keys())
            
            
                
            
        for i in self.ENV:
            end_idx = start_idx + len(self.datasets[i][self.mode]['labels'])
            self.task_idxs[i] = {}
            self.task_idxs[i][self.mode] = [start_idx, end_idx]
            start_idx = end_idx

            for classid in range(self.num_classes):
                sel_idx = np.nonzero(self.datasets[i][self.mode]['labels'] == classid)[0]
                self.class_id_list[classid][i] = sel_idx

            cumulative_batch_size += min(self.mini_batch_size, self.num_examples[i])
            if args.method in ['erm']:
                self.input_dim.append((cumulative_batch_size, 3, 32, 32))
            else:
                self.input_dim.append((min(self.mini_batch_size, self.num_examples[i]), 3, 32, 32))

    def update_historical(self, idx, data_del=False):
        time = self.ENV[idx]
        prev_time = self.ENV[idx - 1]
        #print(time)
        #print(prev_time)
        print(self.mode)
        #print(self.datasets.keys())
        #print(self.datasets[prev_time].keys())
        self.datasets[time][self.mode]['images'] = np.concatenate(
            (self.datasets[time][self.mode]['images'], self.datasets[prev_time][self.mode]['images']), axis=0)
        self.datasets[time][self.mode]['labels'] = np.concatenate(
            (self.datasets[time][self.mode]['labels'], self.datasets[prev_time][self.mode]['labels']), axis=0)
        if data_del:
            del self.datasets[prev_time]
        for classid in range(self.num_classes):
            sel_idx = np.nonzero(self.datasets[time][self.mode]['labels'] == classid)[0]
            self.class_id_list[classid][time] = sel_idx

    def update_historical_K(self, idx, K):
        time = self.ENV[idx]
        prev_time = self.ENV[idx - 1]
        self.window_start = self.ENV[max(0, idx - K)]
        if idx >= K:
            last_K_num_samples = self.input_dim[idx - K][0]
            self.datasets[time][self.mode]['images'] = np.concatenate(
                (self.datasets[time][self.mode]['images'], self.datasets[prev_time][self.mode]['images'][:-last_K_num_samples]), axis=0)
            self.datasets[time][self.mode]['labels'] = np.concatenate(
                (self.datasets[time][self.mode]['labels'], self.datasets[prev_time][self.mode]['labels'][:-last_K_num_samples]), axis=0)
            del self.datasets[prev_time]
            for classid in range(self.num_classes):
                sel_idx = np.nonzero(self.datasets[time][self.mode]['labels'] == classid)[0]
                self.class_id_list[classid][time] = sel_idx
        else:
            self.update_historical(idx)

    def update_current_timestamp(self, time):
        self.current_time = time

    def get_lisa_new_sample(self, time_idx, classid, num_sample):
        # time = self.ENV[time_idx]
        idx_all = self.class_id_list[classid][time_idx]
        sel_idx = np.random.choice(idx_all, num_sample, replace=True)
        image = self.datasets[time_idx][self.mode]['images'][sel_idx]
        label = self.datasets[time_idx][self.mode]['labels'][sel_idx]

        return torch.FloatTensor(image).permute(0, 3, 1, 2).cuda(), torch.LongTensor(label).unsqueeze(1).cuda()

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    def __str__(self):
        return 'yearbook'


class Yearbook(YearbookBase):
    
    def __init__(self, args):
        super().__init__(args=args)

    def __getitem__(self, index):
        if self.args.difficulty and self.mode == 0:
            # Pick a time step from all previous timesteps
            idx = self.ENV.index(self.current_time)
            window = np.arange(0, idx + 1)
            sel_time = self.ENV[np.random.choice(window)]
            start_idx, end_idx = self.task_idxs[sel_time][self.mode]

            # Pick an example in the time step
            sel_idx = np.random.choice(np.arange(start_idx, end_idx))
            index = sel_idx

        image = self.datasets[self.current_time][self.mode]['images'][index]
        label = self.datasets[self.current_time][self.mode]['labels'][index]
        image_tensor = torch.FloatTensor(image).permute(2, 0, 1)
        label_tensor = torch.LongTensor([label])

        if self.args.method in ['simclr', 'swav'] and self.ssl_training:
            tensor_to_PIL = transforms.ToPILImage()
            image_tensor = tensor_to_PIL(image_tensor)
            return image_tensor, label_tensor, ''

        return image_tensor, label_tensor
        
    def __len__(self):
        print(self.datasets.keys())
        print(
            f'mode = {self.mode}'
            f'\t current_time = {self.current_time}'
        )
        return len(self.datasets[self.current_time][self.mode]['labels'])


class YearbookGroup(YearbookBase):
    def __init__(self, args):
        super().__init__(args=args)
        self.num_groups = args.num_groups
        self.group_size = args.group_size
        self.window_end = self.ENV[0]
        self.groupnum = 0

    def __getitem__(self, index):
        if self.mode == 0:
            np.random.seed(index)
            # Select group ID
            idx = self.ENV.index(self.current_time)
            if self.args.non_overlapping:
                possible_groupids = [i for i in range(0, max(1, idx - self.group_size + 1), self.group_size)]
                if len(possible_groupids) == 0:
                    possible_groupids = [np.random.randint(self.group_size)]
            else:
                possible_groupids = [i for i in range(max(1, idx - self.group_size + 1))]
            groupid = np.random.choice(possible_groupids)

            # Pick a time step in the sliding window
            window = np.arange(max(0, idx - groupid - self.group_size), idx + 1)
            sel_time = self.ENV[np.random.choice(window)]
            start_idx, end_idx = self.task_idxs[sel_time][self.mode]

            # Pick an example in the time step
            sel_idx = np.random.choice(np.arange(start_idx, end_idx))
            image = self.datasets[self.current_time][self.mode]['images'][sel_idx]
            label = self.datasets[self.current_time][self.mode]['labels'][sel_idx]

            image_tensor = torch.FloatTensor(image).permute(2, 0, 1)
            label_tensor = torch.LongTensor([label])
            group_tensor = torch.LongTensor([groupid])

            del groupid
            del window
            del sel_time
            del start_idx
            del end_idx
            del sel_idx

            return image_tensor, label_tensor, group_tensor

        else:
            image = self.datasets[self.current_time][self.mode]['images'][index]
            label = self.datasets[self.current_time][self.mode]['labels'][index]
            image_tensor = torch.FloatTensor(image).permute(2, 0, 1)
            label_tensor = torch.LongTensor([label])

            return image_tensor, label_tensor

    def group_counts(self):
        idx = self.ENV.index(self.current_time)
        return torch.LongTensor([1 for _ in range(min(self.num_groups, idx + 1))])

    def __len__(self):
        return len(self.datasets[self.current_time][self.mode]['labels'])
