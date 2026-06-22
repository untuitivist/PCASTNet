
import warnings
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_loader import CustomImageDataset
from function import train_transform, adjust_learning_rate1, adjust_learning_rate2
import stc_datasets
import stc_evaluation
import stc_generation
import stc_training
from stc_utils import clear_cuda_cache

# plt 使用 times new roman 字体
plt.rcParams['font.sans-serif'] = ['Times New Roman']

# 忽略FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)


class STC(object):
    def __init__(self, 
            model_name: str,
            num_classes: int, 
            content_dataset_dir: str,
            style_dataset_dir: str,
            style_transfer_dataset_dir: str,
            content_train_dataset_scale: int = 500,
            content_valid_dataset_scale: int = 0,
            content_test_dataset_scale: int = 0,
            style_train_dataset_scale: int = 50, # [50, 100, 200, 300, 500]
            style_valid_dataset_scale: int = 100,
            style_test_dataset_scale: int = 500,
            test_content_size: int = 512,
            test_style_size: int = 512,
            test_crop: bool = False,
            ): 
        
        self.model_name = model_name

        self.num_classes = num_classes
        
        self.content_dataset_dir = content_dataset_dir
        self.style_dataset_dir = style_dataset_dir
        self.style_transfer_dataset_dir = style_transfer_dataset_dir
        self.content_train_dataset_scale = content_train_dataset_scale
        self.content_valid_dataset_scale = content_valid_dataset_scale
        self.content_test_dataset_scale = content_test_dataset_scale
        self.style_train_dataset_scale = style_train_dataset_scale
        self.style_valid_dataset_scale = style_valid_dataset_scale
        self.style_test_dataset_scale = style_test_dataset_scale

        self.train_transfrom = train_transform()
        self.test_content_tf = train_transform()
        self.test_style_tf = train_transform()
        # self.test_content_tf = test_transform(test_content_size, test_crop)
        # self.test_style_tf = test_transform(test_style_size, test_crop)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        clear_cuda_cache()

        self.n_threads = 0




    def test_classifer_save(self, train_loader, val_loader, test_loader, device, save_dir, i, label):
        return stc_evaluation.save_classifier_checkpoint(
            self, train_loader, val_loader, test_loader, device, save_dir, i, label
        )

    def test_style_transfer_save(self, 
            content_test_images, 
            style_test_images, 
            save_dir, 
            iteration, 
            label
            ):
        return stc_evaluation.save_style_transfer_checkpoint(
            self, content_test_images, style_test_images, save_dir, iteration, label
        )

    def save_feature_map(self, feature, save_path_prefix):
        return stc_evaluation.save_feature_map(feature, save_path_prefix)

    def save_feature_maps(self, contents, styles, save_dir, tag, prefix):
        return stc_evaluation.save_feature_maps(self, contents, styles, save_dir, tag, prefix)


    def build_datasets(self, 
            class_or_transfer: bool,
            test_classes_or_random: bool = True,
            use_train_datasets: list|None = None,
            use_val_datasets: list|None = None,
            use_test_datasets: list|None = None,
            test_style_num: int|None = None,
            test_content_num: int|None = None
            ) -> tuple[CustomImageDataset|tuple[tuple[CustomImageDataset, str]]]:
        return stc_datasets.build_datasets(
            self,
            class_or_transfer,
            test_classes_or_random=test_classes_or_random,
            use_train_datasets=use_train_datasets,
            use_val_datasets=use_val_datasets,
            use_test_datasets=use_test_datasets,
            test_style_num=test_style_num,
            test_content_num=test_content_num,
        )

    def build_classifier_dataloaders(self, 
            use_train_datasets: list,
            use_val_datasets: list,
            use_test_datasets: list,
            batch_size: int = 32
            ) -> tuple[DataLoader]:
        return stc_datasets.build_classifier_dataloaders(
            self,
            use_train_datasets,
            use_val_datasets,
            use_test_datasets,
            batch_size,
        )
    
    def build_style_transfer_iters(self, 
            batch_size: int = 32,
            test_classes_or_random: bool = True,
            test_content_num: int = 10,
            test_style_num: int = 10,
            ) -> tuple[iter, iter, list, list]:
        return stc_datasets.build_style_transfer_iters(
            self,
            batch_size=batch_size,
            test_classes_or_random=test_classes_or_random,
            test_content_num=test_content_num,
            test_style_num=test_style_num,
        )

    def train_classifier(self, 
            max_iter: int = 1000,
            save_iter: int = 1e25,
            save_dir: str = f'./experiments{time.strftime("/%m-%d-%H-%M", time.localtime(time.time()))}',
            log_dir: str = f'./logs{time.strftime("/%m-%d-%H-%M", time.localtime(time.time()))}',
            lr: float = 1e-4,
            lr_decay: float = 5e-5,
            batch_size: int = 32,
            use_train_datasets: list = ['content', 'style'], # 'content' or 'style' or 'style_transfer'
            use_val_datasets: list = ['style'], # 'content' or 'style'
            use_test_datasets: list = ['style'], # 'content' or 'style'
            preheat: int = 10,
            realy_stop: int = 100,
            freeze_encoder: bool = False,
            freeze_classifier: bool = False,
            ):
        

        # Setting
        device = self.device
        # Save setting于save_dir/setting.json
        setting = {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'content_dataset_dir': self.content_dataset_dir,
            'style_dataset_dir': self.style_dataset_dir,
            'style_transfer_dataset_dir': self.style_transfer_dataset_dir,
            'style_train_dataset_scale': self.style_train_dataset_scale,
            'n_threads': self.n_threads,
            'device': str(device),
            'max_iter': max_iter,
            'lr': lr,
            'lr_decay': lr_decay,
            'batch_size': batch_size,
            'use_train_datasets': use_train_datasets,
            'use_val_datasets': use_val_datasets,
            'use_test_datasets': use_test_datasets,
            'preheat': preheat,
            'realy_stop': realy_stop,
            'freeze_encoder': freeze_encoder,
            'freeze_classifier': freeze_classifier,            
        }
        # 创建setting.json文件
        save_dir, log_dir, writer = stc_training.prepare_stage_output(self, save_dir, 'c', setting)

        
        # Define network
        network = self.net
        network.train()
        network.to(device)
        print('[+]Define the full network')

        # Freeze encoder
        if freeze_encoder:
            stc_training.freeze_module(network.vgg, 'encoder')
        
        # Freeze classifier
        if freeze_classifier:
            stc_training.freeze_module(network.classifier, 'classifier')

        # Define optimizer
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, network.parameters()), lr=lr)

        # Define loss function
        criterion = nn.CrossEntropyLoss()

        # Create data loaders
        train_loader, val_loader, test_loader = \
            self.build_classifier_dataloaders(use_train_datasets,
                                              use_val_datasets,
                                              use_test_datasets,
                                              batch_size)

        print('[+]Start training...')
        best_val_acc = 0.0
        best_epoch_num = float('inf')
        try:
            for i in range(max_iter):
                adjust_learning_rate2(lr, lr_decay, optimizer, iteration_count=i)
                
                # Train
                network.train()
                train_loss = 0.0
                for images, labels in tqdm(train_loader, total=len(train_loader), desc='Processing train segments', ncols=100, leave=False):
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    outputs = network(images)
                    loss = criterion(outputs, labels)
                    train_loss += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    writer.add_scalar('train_loss_step', loss.item(), i)

                    # break  # For demonstration, break after one batch

                train_loss /= len(train_loader)
                writer.add_scalar('train_loss', train_loss, i)

                # Valid
                network.eval()
                val_loss = 0
                correct = 0
                total = 0
                with torch.no_grad():
                    for images, labels in tqdm(val_loader, total=len(val_loader), desc='Processing valid segments', ncols=100, leave=False):
                        images = images.to(device)
                        labels = labels.to(device)
                        
                        outputs = network(images)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                val_loss /= len(val_loader)
                val_acc = 100 * correct / total
                writer.add_scalar('val_loss', val_loss, i)
                writer.add_scalar('val_acc', val_acc, i)

                print(f'Iteration {i}, \tTrain Loss: {train_loss:.4f}, \tValid Loss: {val_loss:.4f}, \tValid Accuracy: {val_acc:.2f}%')

                # Save and test
                if val_acc > best_val_acc:
                    if i > preheat: 
                        best_val_acc = val_acc
                        best_epoch_num = i
                        print('[+]Best Valid Accuracy...')
                        self.test_classifer_save(train_loader, val_loader, test_loader, device, save_dir, i, 'best')
                elif ((i + 1) == max_iter) or (i == best_epoch_num + realy_stop):
                    print('[+]Stopping training...')
                    print('[+]Saving last models...')
                    self.test_classifer_save(train_loader, val_loader, test_loader, device, save_dir, i, 'last') 
                    break
                if (i + 1) % save_iter == 0:
                    print('[+]Saving models...')
                    self.test_classifer_save(train_loader, val_loader, test_loader, device, save_dir, i, 'step')
        # Ctrl+C Interrupt
        except KeyboardInterrupt:
            print('[+]KeyboardInterrupt: Stopping training...')
            print('[+]Saving last models...')
            self.test_classifer_save(train_loader, val_loader, test_loader, device, save_dir, i, 'last')

        finally:
            writer.close()
            del optimizer, criterion, train_loader, val_loader, test_loader, writer
            clear_cuda_cache()

    def train_style_transfer(self,
            max_iter: int = 10000,
            save_dir: str = f'./experiments{time.strftime("/%m-%d-%H-%M", time.localtime(time.time()))}',
            # log_dir: str = f'./logs{time.strftime("/%m-%d-%H-%M", time.localtime(time.time()))}',
            lr: float = 1e-4,
            lr_decay: float = 5e-5,
            batch_size: int = 16,
            content_weight: float = 10.0,
            style_weight: float = 5.0,
            perceptual_weight: float = 0,
            totalvariation_weight: float = 0.0,
            energe_weight: float = 5.0,
            test_classes_or_random: bool = True,
            test_content_num: int = 10,
            test_style_num: int = 10,
            preheat: int = 200,
            realy_stop: int = 200,
            freeze_encoder: bool = True,
            freeze_adailn: bool = False,
            freeze_decoder: bool = False,
            ):
        
        # Setting
        device = self.device
        
        # Save setting于save_dir/setting.json
        setting = {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'content_dataset_dir': self.content_dataset_dir,
            'style_dataset_dir': self.style_dataset_dir,
            'style_transfer_dataset_dir': self.style_transfer_dataset_dir,
            'style_train_dataset_scale': self.style_train_dataset_scale,
            'n_threads': self.n_threads,
            'device': str(device),
            'max_iter': max_iter,
            'lr': lr,
            'lr_decay': lr_decay,
            'batch_size': batch_size,
            'content_weight': content_weight,
            'style_weight': style_weight,
            'perceptual_weight': perceptual_weight,
            'totalvariation_weight': totalvariation_weight,
            'energe_weight': energe_weight,
            'test_classes_or_random': test_classes_or_random,
            'test_content_num': test_content_num,
            'test_style_num': test_style_num,  
            'preheat': preheat,
            'realy_stop': realy_stop,
            'freeze_encoder': freeze_encoder,
            'freeze_adailn': freeze_adailn,
            'freeze_decoder': freeze_decoder,     
        }
        # 创建setting.json文件
        save_dir, log_dir, writer = stc_training.prepare_stage_output(self, save_dir, 'st', setting, indent=4)

        # Define network
        network = self.net
        network.train()
        network.to(device)
        print('[+]Define the full network')

        # Define optimizer
        optimizer = torch.optim.Adam(network.decoder.parameters(), lr=lr)

        # Create data iters
        content_iter, style_iter, content_test_images, style_test_images = \
            self.build_style_transfer_iters(batch_size, test_classes_or_random, test_content_num, test_style_num)

        # Freeze encoder
        if freeze_encoder:
            stc_training.freeze_module(network.encoder, 'encoder')

        # Freeze adailn
        if freeze_adailn:
            stc_training.freeze_module(network.adailn, 'adailn')
        
        # Freeze decoder
        if freeze_decoder:
            stc_training.freeze_module(network.decoder, 'decoder')

        print('[+]Start training...')
        best_loss = float('inf')
        best_epoch_num = float('inf')
        try:
            for i in range(max_iter):
                adjust_learning_rate1(lr, lr_decay, optimizer, iteration_count=i)
                content_images, _ = next(content_iter)
                style_images, _ = next(style_iter)
                content_images = content_images.to(device)
                style_images = style_images.to(device)
                _, loss_c, loss_s, loss_p, loss_v, loss_e = network(content_images, style_images)
                loss_c = content_weight * loss_c
                loss_s = style_weight * loss_s
                loss_p = perceptual_weight * loss_p
                loss_v = totalvariation_weight * loss_v
                loss_e = energe_weight * loss_e
                loss = loss_c + loss_s + loss_p + loss_v + loss_e

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                writer.add_scalar('loss_content', loss_c.item(), i)
                writer.add_scalar('loss_style', loss_s.item(), i)
                writer.add_scalar('loss_perceptual', loss_p.item(), i)
                writer.add_scalar('loss_total_variation', loss_v.item(), i)
                writer.add_scalar('loss_energe', loss_e.item(), i)

                print(f'Iteration {i},\tContent Loss: {loss_c.item():.4f},\tStyle Loss: {loss_s.item():.4f},\tPerceptual Loss: {loss_p.item():.4f},\tTotal Variation Loss: {loss_v.item():.4f},\tEnerge Loss: {loss_e.item():.4f}')

                if loss < best_loss:
                    if i > preheat:
                        best_loss = loss
                        best_epoch_num = i
                        print('[+]Best Loss...')
                        self.test_style_transfer_save(content_test_images, style_test_images, save_dir, i, 'best')
                        # 保存experiments\content_feature_map_0.png
                        # experiments\content_feature_map_1.png
                        # experiments\content_feature_map_2.png
                        # experiments\style_feature_map_0.png
                        # experiments\style_feature_map_1.png
                        # experiments\style_feature_map_2.png
                        # experiments\transfer_feature_map_0.png
                        # experiments\transfer_feature_map_1.png
                        # experiments\transfer_feature_map_2.png
                        self.save_feature_maps(content_test_images, style_test_images, save_dir, i, 'best')


                if (i + 1) == max_iter or (i == best_epoch_num + realy_stop):
                    print('[+]Stopping training...')
                    print('[+]Saving last models...')
                    self.test_style_transfer_save(content_test_images, style_test_images, save_dir, i, 'last')
                    break
        # Ctrl+C Interrupt
        except KeyboardInterrupt:
            print('[+]KeyboardInterrupt: Stopping training...')
            print('[+]Saving last models...')
            self.test_style_transfer_save(content_test_images, style_test_images, save_dir, i, 'last')

        finally:
            writer.close()
            del optimizer, content_iter, style_iter, writer, content_test_images, style_test_images
            clear_cuda_cache()

    def make_adailn_dataset(self, folder_label):
        return stc_generation.make_adailn_dataset(self, folder_label)


