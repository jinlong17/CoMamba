import argparse
import os
import statistics

import sys
# sys.path.append("/home/jinlong/4.3D_detection/Noise_V2V/v2vreal")
# sys.path.remove("/home/jinlong/1.Detection_Sets/V2V4Real")
# print(sys.path)



import torch
import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.models.point_pillar_V2VAM_LCRN import fusion_module, LCRN_module

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--model', default='',
                        help='for fine-tuned training path')
    parser.add_argument('--LCRN_module', default='',
                        help='upload the trained LCRN moulde from path')
    parser.add_argument("--half", action='store_true', help="whether train with half precision")
    parser.add_argument('--isSim', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    print('Dataset Building')
    print(hypes["name"], "is loaded!!!!")
    # print(hypes)

    run = False
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False
                                              )

    train_loader = DataLoader(opencood_train_dataset,
                              batch_size=hypes['train_params']['batch_size'],
                              num_workers=8,
                              collate_fn=opencood_train_dataset.collate_batch_train,
                              shuffle=True,
                              pin_memory=False,
                              drop_last=True)
    val_loader = DataLoader(opencood_validate_dataset,
                            batch_size=hypes['train_params']['batch_size'],
                            num_workers=8,
                            collate_fn=opencood_train_dataset.collate_batch_train,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=True)


    '''
    point pillar opv2v module
    '''
    print('Creating Point Pillar opv2v module')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)

    '''
    fusion module
    '''
    print('Creating fusion module')
    fusion = fusion_module(hypes['model']['args'])
    if torch.cuda.is_available():
        fusion.to(device)

    
    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)
    optimizer_fusion = train_utils.setup_optimizer(hypes, fusion)##fusion module
    # lr scheduler setup
    num_steps = len(train_loader)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)
    scheduler_fusion = train_utils.setup_lr_schedular(hypes, optimizer_fusion, num_steps)##fusion module


    '''
    LCRN module
    '''
    LCRN = LCRN_module()
    if torch.cuda.is_available():
        LCRN.to(device)
    print("Created LCRN model!")
    optimizer_LCRN = train_utils.setup_optimizer(hypes, LCRN)
    scheduler_LCRN = train_utils.setup_lr_schedular(hypes, optimizer_LCRN, num_steps)




    if opt.LCRN_module:
        LCRN_path = opt.LCRN_module
        # LCRN_state = torch.load(os.path.join(LCRN_path,'latest.pth'))
        LCRN_state = torch.load(LCRN_path)
        # LCRN.load_state_dict(LCRN_state['LCRN_state_dict'])

        LCRN_dict = LCRN.state_dict()
        LCRN_state = {k: v for k, v in LCRN_state.items() if (k in LCRN_dict and v.shape == LCRN_dict[k].shape)}
        LCRN_dict.update(LCRN_state)
        LCRN.load_state_dict(LCRN_dict)
        print('Loaded pretrained LCRN model from {}'.format(LCRN_path))


    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        print('Loaded model from {}'.format(saved_path))

    else:
        if opt.model:
            saved_path = train_utils.setup_train(hypes)
            '''
            load the pretrained 3D detection model
            '''
            model_path = opt.model
            init_epoch = 0
            pretrained_state = torch.load(model_path)

            
            # model.load_state_dict(pretrained_state['model_state_dict'])
            ############
            model_dict = model.state_dict()
            pretrained_state = {k: v for k, v in pretrained_state.items() if (k in model_dict and v.shape == model_dict[k].shape)}
            model_dict.update(pretrained_state)
            model.load_state_dict(model_dict)
            print('Loaded pretrained Backbone model from {}'.format(model_path))   

            '''
            load the pretrained fusion module
            ''' 

            fusion_path = opt.model
            init_epoch = 0
            pretrained_state = torch.load(fusion_path)

            # fusion.load_state_dict(pretrained_state['fusion_state_dict'])
            ############
            model_dict = fusion.state_dict()
            pretrained_state = {k: v for k, v in pretrained_state.items() if (k in model_dict and v.shape == model_dict[k].shape)}
            model_dict.update(pretrained_state)
            fusion.load_state_dict(model_dict)
            print('Loaded pretrained Fusion model from {}'.format(model_path))  
        else:
            init_epoch = 0
            # if we train the model from scratch, we need to create a folder
            # to save the model,
            saved_path = train_utils.setup_train(hypes)

    # record training
    writer = SummaryWriter(saved_path)

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    txt_path = os.path.join(saved_path, 'training_eval_log.txt')
    txt_log = open(txt_path, "w")

    print('Training start')
    epoches = hypes['train_params']['epoches']
    # used to help schedule learning rate
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step(epoch)
        if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
            scheduler.step_update(epoch * num_steps + 0)
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)


        for i, batch_data in enumerate(train_loader):


            batch_data = train_utils.to_device(batch_data, device)

            # the model will be evaluation mode during validation
            model.train()
            fusion.train()
            LCRN.train()

            # point pillar module forward

            ouput_dict = model(batch_data['ego'])

            #LCRN module forward
            spatial_features_2d_repaired = LCRN(ouput_dict)

            #fusion module forward
            final_output = fusion(spatial_features_2d_repaired,ouput_dict)



            model.zero_grad()
            optimizer.zero_grad()
            fusion.zero_grad()
            optimizer_fusion.zero_grad()
            final_loss = criterion(final_output, batch_data['ego']['label_dict'])

            LCRN.zero_grad()
            optimizer_LCRN.zero_grad()
            LCRN_loss = final_output['L1_diff_loss_computed'].requires_grad_()


            total_loss = final_loss
            # total_loss = final_loss
            total_loss.backward()
            optimizer.step()
            optimizer_fusion.step()
            optimizer_LCRN.step()

            
            # criterion.logging(epoch, i, len(train_loader), writer, pbar=pbar2)
            criterion.logging_LCRN(epoch, i, len(train_loader), writer, final_output,run, pbar=pbar2)
            pbar2.update(1)


            if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
                scheduler.step_update(epoch * num_steps + i)

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    model.eval()
                    LCRN.eval()
                    fusion.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    ouput_dict = model(batch_data['ego'])
                    spatial_features_2d_repaired = LCRN(ouput_dict)
                    final_output = fusion(spatial_features_2d_repaired,ouput_dict)

                    final_loss = criterion(final_output,
                                           batch_data['ego']['label_dict'])
                    valid_ave_loss.append(final_loss.item())
            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                               valid_ave_loss))
                                                            
            txt_log.write('At epoch' + str(epoch+1)+',  the validation loss is'+ str(valid_ave_loss) + 'save in '+ str(os.path.join(saved_path,'net_epoch%d.pth' % (epoch + 1))) + '\n')

            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

        if epoch % hypes['train_params']['save_freq'] == 0:

            torch.save({
                        'model_state_dict': model.state_dict(),
                        'fusion_state_dict': fusion.state_dict(),
                        'LCRN_state_dict': LCRN.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'optimizer_fusion_state_dict': optimizer_fusion.state_dict(),
                        'optimizer_LCRN_state_dict': optimizer_LCRN.state_dict()
                        }, os.path.join(saved_path,'net_epoch%d.pth' % (epoch + 1)))                 


    print('Training Finished, checkpoints saved to %s' % saved_path)
    txt_log.close()


if __name__ == '__main__':
    main()
