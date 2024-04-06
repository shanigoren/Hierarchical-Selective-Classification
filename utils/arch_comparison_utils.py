import os

import numpy as np
import timm
import torch
from torchvision import transforms
from .data_utils import load_model_results, save_model_results
from .log_utils import Logger
import torchvision.transforms as tvtf
import timm.models.resnetv2 as timm_bit
import timm.models.resnet as timm_resnet
import torchvision.models as torchvision_models
import clip


def normalize(a):
    a = np.array(a)
    a = (a - np.mean(a)) / np.std(a)
    return a


def to_cpu(tensor):
    return tensor.detach().cpu().numpy()


def get_fc_layer(model):
    """
    the worst thing i've ever written :)
    :param model: trained model
    :return: the weights of the classification layer in numpy
    """
    if hasattr(model, 'fc'):
        if isinstance(model.fc, torch.nn.Linear):
            return model.fc.weight.detach().clone().cpu().numpy()
        if isinstance(model.fc, torch.nn.Conv2d) and model.fc.kernel_size == (1, 1):
            return model.fc.weight.detach().clone().cpu().numpy()

    if hasattr(model, 'last_linear'):
        if isinstance(model.last_linear, torch.nn.Linear):
            return model.last_linear.weight.detach().clone().cpu().numpy()

    if hasattr(model, 'classifier'):
        if isinstance(model.classifier, torch.nn.Linear):
            return model.classifier.weight.detach().clone().cpu().numpy()

        if isinstance(model.classifier, torch.nn.Conv2d) and model.classifier.kernel_size == (1, 1):
            return model.classifier.weight.detach().clone().cpu().numpy()

        if isinstance(model.classifier, torch.nn.Sequential):
            if isinstance(model, torchvision_models.SqueezeNet):
                return model.classifier[1].weight.detach().clone().cpu().numpy()

            return model.classifier[-1].weight.detach().clone().cpu().numpy()

        if hasattr(model.classifier, 'fc'):
            if isinstance(model.classifier.fc, torch.nn.Linear):
                return model.classifier.fc.detach().clone().cpu().numpy()

    if hasattr(model, 'head'):
        if isinstance(model.head, torch.nn.Linear):
            return model.head.weight.detach().clone().cpu().numpy()
        if hasattr(model.head, 'fc'):

            if isinstance(model.head.fc, torch.nn.Linear):
                return model.head.fc.detach().clone().cpu().numpy()

            if isinstance(model.head.fc, torch.nn.Conv2d) and model.head.fc.kernel_size == (1, 1):
                return model.head.fc.weight.detach().clone().cpu().numpy()

        if hasattr(model.head, 'l'):
            if isinstance(model.head.l, torch.nn.Linear):
                return model.head.l.detach().clone().cpu().numpy()

    if hasattr(model, 'classif'):
        if isinstance(model.classif, torch.nn.Linear):
            return model.classif.weight.detach().clone().cpu().numpy()

    return False


def log_ood_results(model_info, results, severity_type, confidence_metric, percentiles, version=1):
    global_log_path = f'./results_v{version}/global_{severity_type}_{confidence_metric}.csv'
    headers = list(model_info.keys())
    metrics_headers = [f's {p :.2f}_' + k for p in percentiles for k in results[0].keys()]
    headers += metrics_headers
    global_logger = Logger(global_log_path, headers, overwrite=False)
    data = {f's {p :.2f}_' + k: results[i][k] for i, p in enumerate(percentiles) for k in results[0].keys()}
    data.update(model_info)
    model_name = model_info['model_name']
    global_logger.log(data)

    model_dir = os.path.join(RESULTS_AND_STATS_BASE_FOLDER, model_name)
    private_log_path = os.path.join(model_dir, f'{model_name}_{severity_type}_{confidence_metric}.csv')
    headers = ['percentile'] + [k for k in results[0].keys()]
    private_log = Logger(private_log_path, headers, overwrite=True)
    for i, r in enumerate(results):
        r['percentile'] = percentiles[i]
        private_log.log(r)


def aggregate_confidences(results_list):
    confidences = {k: [] for k in results_list[0].keys()}

    for r in results_list:
        for k, v in confidences.items():
            v.extend(r[k])

    confidences = {k: np.concatenate(v) for k, v in confidences.items()}
    return confidences


def fix(results):
    results_list = [results]
    confidences = aggregate_confidences(results_list)
    return confidences


def fix_model(model_name):
    name_2_load = 'stats_mcp_entropy_all_val_0'
    res = load_model_results(model_name, name_2_load)
    result = fix(res)
    assert len(result['labels']) / 50 == 15293
    save_name = 'stats_mcp_entropy_all_val'
    save_model_results(model_name, result, save_name)


def gather_results(model_name, world_size, tag):
    results_list = []
    for r in range(world_size):
        res = load_model_results(model_name, f'{tag}_{r}')
        assert res is not None
        results_list.append(res)

    return results_list


def get_embedding_size(model_name):
    # model = timm.create_model(model_name, pretrained=False)
    # return model.num_features
    # turn on when you finally add torchvision integration
    model = create_model_and_transforms(args, model_name, False)
    try:
        return model.num_features
    except Exception:
        return get_fc_layer(model).shape[1]


def translate_model_name(model_name, to_our_convention=False):
    """"
    translates from timm convention to our convention or vise-versa depending on to_torchvision value
    our convention adds '_torchvision' suffix to  torchvision models and removes 'tv_' prefix if exists.
    """
    torchvision_duplicated_models_in_timms = ['resnext101_32x8d', 'tv_densenet121', 'tv_resnet101', 'tv_resnet152',
                                              'tv_resnet34', 'tv_resnet50', 'tv_resnext50_32x4d', 'vgg11', 'vgg11_bn',
                                              'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',
                                              'wide_resnet101_2', 'densenet169', 'densenet201', 'densenet161',
                                              'inception_v3', 'resnet18']
    if to_our_convention:
        if model_name in torchvision_duplicated_models_in_timms:
            return model_name.replace('tv_', '') + '_torchvision'
        return model_name
    else:
        if model_name.replace('_torchvision', '') in torchvision_duplicated_models_in_timms:
            return model_name.replace('_torchvision', '')
        if 'tv_' + model_name.replace('_torchvision', '') in torchvision_duplicated_models_in_timms:
            return 'tv_' + model_name.replace('_torchvision', '')
        return model_name


def translate_models_names(models_names, to_our_convention=False):
    models_names_translated = [translate_model_name(model_name, to_our_convention) for model_name in models_names]
    return models_names_translated


def get_models_list():
    models_names = timm_lib.models.list_models(pretrained=True, exclude_filters=['*_in21k', '*_in22k'])

    # Duplicated models as of August 2021:
    torchvision_duplicated_models_in_timms = ['resnext101_32x8d', 'tv_densenet121', 'tv_resnet101', 'tv_resnet152',
                                          'tv_resnet34', 'tv_resnet50', 'tv_resnext50_32x4d', 'vgg11', 'vgg11_bn',
                                          'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',
                                              'wide_resnet101_2', 'densenet169', 'densenet201', 'densenet161',
                                              'inception_v3', 'resnet18']
    models_names = [model_name for model_name in models_names if model_name not in torchvision_duplicated_models_in_timms]
    # Adding Torchvision models:
    # TODO: Check the resnext101_32x8d_torchvision model thingy
    # Removed 'resnext101_32x8d_torchvision' because it's the same as timm's
    models_names.extend(['alexnet_torchvision', 'vgg11_torchvision', 'vgg11_bn_torchvision', 'vgg13_torchvision',
                         'vgg13_bn_torchvision', 'vgg16_torchvision', 'vgg16_bn_torchvision', 'vgg19_torchvision',
                         'vgg19_bn_torchvision', 'resnet18_torchvision', 'resnet34_torchvision', 'resnet50_torchvision',
                         'resnet101_torchvision', 'resnet152_torchvision', 'squeezenet1_0_torchvision',
                         'squeezenet1_1_torchvision', 'densenet121_torchvision', 'densenet169_torchvision',
                         'densenet161_torchvision', 'densenet201_torchvision', 'inception_v3_torchvision',
                         'googlenet_torchvision', 'shufflenet_v2_x0_5_torchvision', 'shufflenet_v2_x1_0_torchvision',
                         'mobilenet_v2_torchvision', 'mobilenet_v3_large_torchvision', 'mobilenet_v3_small_torchvision',
                         'resnext50_32x4d_torchvision', 'resnext101_32x8d_torchvision', 'wide_resnet50_2_torchvision',
                         'wide_resnet101_2_torchvision', 'mnasnet0_5_torchvision', 'mnasnet1_0_torchvision'])
    # New torchvision models. I think I've filtered those comming from timm's repo:
    models_names.extend(['regnet_x_16gf_torchvision', 'regnet_x_32gf_torchvision', 'regnet_y_400mf_torchvision', 'regnet_y_800mf_torchvision', 'regnet_y_1_6gf_torchvision', 'regnet_y_3_2gf_torchvision', 'regnet_y_8gf_torchvision', 'regnet_y_16gf_torchvision', 'regnet_y_32gf_torchvision'])
    models_names.extend(['regnet_x_8gf_torchvision', 'vit_b_16_torchvision', 'vit_b_32_torchvision', 'vit_l_16_torchvision', 'vit_l_32_torchvision', 'convnext_tiny_torchvision', 'convnext_large_torchvision', 'convnext_small_torchvision', 'convnext_base_torchvision'])
    models_names.extend(['regnet_x_400mf_torchvision', 'regnet_x_800mf_torchvision', 'regnet_x_1_6gf_torchvision', 'regnet_x_3_2gf_torchvision'])
    # Adding SWAG models (they will be incorporated into the torchvision library in 14.0)
    models_names.extend(['vit_b16_in1k_facebookSWAG', 'vit_l16_in1k_facebookSWAG', 'vit_h16_in1k_facebookSWAG', 'regnety_16gf_in1k_facebookSWAG', 'regnety_32gf_in1k_facebookSWAG', 'regnety_128gf_in1k_facebookSWAG'])
    # Removing DINO models
    models_names = [model_name for model_name in models_names if '_dino' not in model_name]
    # Removing SWINV2 models for 21k and not finetuned that don't follow the "in22k" convention
    models_names = [model_name for model_name in models_names if not ('swinv2' in model_name and model_name.endswith('22k'))]
    for model_name in models_names:
        assert not model_name.startswith('tv_')
        assert not model_name.endswith('_dino')

    # Special models to add:
    # CLIP models
    clip_models_names = ['CLIP_RN50', 'CLIP_RN101', 'CLIP_RN50x4', 'CLIP_RN50x16', 'CLIP_RN50x64', 'CLIP_ViT-B~32', 'CLIP_ViT-B~16', 'CLIP_ViT-L~14', 'CLIP_ViT-L~14@336px']
    models_names.extend(clip_models_names)
    # CLIP - finetuned models
    # clip_models_names = ['CLIP_finetuned_RN50', 'CLIP_finetuned_RN101', 'CLIP_finetuned_RN50x4', 'CLIP_finetuned_RN50x16', 'CLIP_finetuned_RN50x64', 'CLIP_finetuned_ViT-B~32', 'CLIP_finetuned_ViT-B~16', 'CLIP_finetuned_ViT-L~14', 'CLIP_finetuned_ViT-L~14@336px']
    # models_names.extend(clip_models_names)
    # Original ViTs from the original paper:
    vits = old_timm_lib.timm.models.list_models(pretrained=True, exclude_filters=['*_in21k', '*_in22k'])
    vits = [vit for vit in vits if 'vit' in vit and 'deit' not in vit and 'miil' not in vit]
    vits = [f'{vit}_original' for vit in vits]
    models_names.extend(vits)

    # Forgotten 2nd smaller student of BiT (160x160 student of a 224x224 teacher):
    models_names.extend(['resnetv2_50x1_bit_distilled_160_from224teacher', ])
    # Forgotten TNT model (transformer in transformer) not included in timm's models list:
    models_names.extend(['tnt_s_patch16_224', ])

    return models_names


def get_vit_list():
    models_names = get_models_list()
    models_names = [model_name for model_name in models_names if 'deit' in model_name or ('vit' in model_name and
                                                                                           'levit' not in model_name and
                                                                                           'CLIP' not in model_name and
                                                                                           'convit' not in model_name and
                                                                                           'crossvit' not in model_name and
                                                                                           'mobile' not in model_name)]
    return models_names


def get_clip_list():
    models_names = ['CLIP_RN50', 'CLIP_RN101', 'CLIP_RN50x4', 'CLIP_RN50x16', 'CLIP_RN50x64', 'CLIP_ViT-B~32', 'CLIP_ViT-B~16', 'CLIP_ViT-L~14', 'CLIP_ViT-L~14@336px']
    # CLIP - finetuned models
    # Following is a partial list. RN50 and ViT-B-32 is missing until their finetuning is done.
    clip_models_names = ['CLIP_finetuned_RN101', 'CLIP_finetuned_RN50x4', 'CLIP_finetuned_RN50x16', 'CLIP_finetuned_RN50x64', 'CLIP_finetuned_ViT-B~16', 'CLIP_finetuned_ViT-L~14', 'CLIP_finetuned_ViT-L~14@336px']
    # clip_models_names = ['CLIP_finetuned_RN50', 'CLIP_finetuned_RN101', 'CLIP_finetuned_RN50x4', 'CLIP_finetuned_RN50x16', 'CLIP_finetuned_RN50x64', 'CLIP_finetuned_ViT-B~32', 'CLIP_finetuned_ViT-B~16', 'CLIP_finetuned_ViT-L~14', 'CLIP_finetuned_ViT-L~14@336px']
    models_names.extend(clip_models_names)
    return models_names


def _create_resnetv2_distilled_160_from224teacher(variant, pretrained=True, **kwargs):
    feature_cfg = dict(flatten_sequential=True)
    return timm_bit.build_model_with_cfg(
        timm_bit.ResNetV2, variant, pretrained,
        default_cfg=timm_bit._cfg(
            url='https://storage.googleapis.com/bit_models/distill/R50x1_160.npz',
            input_size=(3, 160, 160), interpolation='bicubic'),
        feature_cfg=feature_cfg,
        pretrained_custom_load=True,
        **kwargs)


def _create_resnet50_pruned(variant, pretrained, **kwargs):
    assert variant in [70, 83, 85]
    cfg = timm_resnet._cfg(
        url=f'https://degirum-model-checkpoints.s3.amazonaws.com/pruned_models/resnet50_pruned_{variant}_state_dict.pth')
    return timm_resnet.build_model_with_cfg(
        timm_resnet.ResNet, variant='resnet50', pretrained=pretrained,
        default_cfg=cfg,
        **kwargs)


default_transform = tvtf.Compose([tvtf.Resize(256), tvtf.CenterCrop(224), tvtf.ToTensor(),
                                  tvtf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def create_model_and_transforms_OOD(model_name, pretrained=True):
    model, transforms_ = create_model_and_transforms(args, model_name, pretrained)
    open_img_transforms = get_open_img_transforms()
    transforms = tvtf.Compose([open_img_transforms, transforms_])

    return model, transforms


def create_model_and_transforms(args, model_name, pretrained=True, models_dir='./timmResNets', weights_generic_name='model_best.pth.tar'):    
    if '_torchvision' in model_name:
        pretrained_str = f'{pretrained}'

        architecture = model_name.replace('_torchvision', '')
        if 'MCdropout' in model_name:
            architecture = architecture.replace('_MCdropout', '')  # Since it's the same torchvision model
        if '_quantized' in model_name:
            architecture = architecture.replace('_quantized', '')
            model = eval(
                f'torchvision_models.quantization.' + architecture + f'(pretrained={pretrained_str}, quantize=True).eval().cuda()')
        else:
            try:
                model = eval(f'torchvision_models.' + architecture + f'(pretrained={pretrained_str}).eval().cuda()')
            except Exception:
                raise Exception(f'Failed to create torchvision model: {architecture}')

        if architecture == 'inception_v3':
            transform = tvtf.Compose([tvtf.Resize(342), tvtf.CenterCrop(299), tvtf.ToTensor(),
                                      tvtf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        elif 'convnext' in architecture:
            resize = 232
            if architecture == 'convnext_small':
                resize = 230
            elif architecture == 'convnext_tiny':
                resize = 236
            transform = tvtf.Compose([tvtf.Resize(resize), tvtf.CenterCrop(224), tvtf.ToTensor(),
                                              tvtf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:
            transform = default_transform
    elif model_name == 'resnetv2_50x1_bit_distilled_160_from224teacher':
        model = _create_resnetv2_distilled_160_from224teacher('resnetv2_50x1_bit_distilled_160_from224teacher',
                                                              pretrained=pretrained, stem_type='fixed',
                                                              conv_layer=timm_bit.partial(timm_bit.StdConv2d, eps=1e-8),
                                                              layers=[3, 4, 6, 3], width_factor=1).eval().cuda()
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
    elif 'resnet50_pruned' in model_name:  # It's a pruned resnet50 model
        prune_level = [int(s) for s in model_name.split('_') if s.isdigit()]
        assert len(prune_level) == 1
        model = _create_resnet50_pruned(prune_level[0], pretrained,
                                        **dict(block=timm_resnet.Bottleneck, layers=[3, 4, 6, 3])).eval().cuda()
        transform = default_transform
    elif 'tnt_s_patch16_224' in model_name:
        from timm_lib.timm.models.tnt import tnt_s_patch16_224
        model = tnt_s_patch16_224(pretrained=pretrained).eval().cuda()
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
    elif 'resnet50_seed' in model_name:
        checkpoint_path = f'{models_dir}/{model_name.split("_")[1]}/{weights_generic_name}'
        # model = timm_lib.create_model('resnet50', pretrained=False, checkpoint_path=checkpoint_path)
        model = timm_lib.create_model('resnet50', pretrained=False)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model = model.eval().cuda()
        # Creating the model specific data transformation
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
    elif 'CLIP' in model_name:
        architecture = model_name.replace('CLIP_', '')
        architecture = architecture.replace('hCLIP_', '')
        if 'finetuned' in model_name:
            architecture = architecture.replace('finetuned_', '')
            finetuned = True
        else:
            finetuned = False
        architecture = architecture.replace('~', '/')
        model, transform = clip.load(architecture, device="cuda")
        from utils.clip_utils import ImageNetClip
        model = ImageNetClip(model, preprocess=transform, linear_probe=finetuned, name=model_name, labels=args.labels)
    elif 'facebookSWAG' in model_name:
        architecture = model_name.replace('_facebookSWAG', '')
        model = torch.hub.load("facebookresearch/swag", model=architecture).eval().cuda()
        resize = 384
        if ('vit_l16_in1k' in model_name) or ('vit_h14_in1k' in model_name):
            resize = 512
        transform = tvtf.Compose([tvtf.Resize(resize, interpolation=tvtf.InterpolationMode.BICUBIC), tvtf.CenterCrop(resize),
                                  tvtf.ToTensor(), tvtf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    elif 'vit' in model_name and 'original' in model_name:
        architecture = model_name.replace('_original', '')
        model = old_timm_lib.timm.models.create_model(architecture, pretrained=pretrained).eval().cuda()
        # Creating the model specific data transformation
        config = old_resolve_data_config({}, model=model)
        transform = old_create_transform(**config)
    elif args.inat:
        if model_name.endswith('.pth'):
            architecture = model_name.split('_iNat21-mini')[0]
            model = timm.create_model(architecture, pretrained=True)
            data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
            transform = timm.data.create_transform(**data_cfg)
            checkpoint_path = f'resources/inat_ft_models/{model_name}'
            checkpoint = torch.load(checkpoint_path)
            num_classes = 10000
            if hasattr(model, 'fc'):
                model.fc = torch.nn.Linear(model.fc.in_features, num_classes)           
            elif hasattr(model, 'classifier'):
                model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
            elif hasattr(model, 'head'):
                model.head = torch.nn.Linear(model.head.in_features, num_classes)
            model.load_state_dict(checkpoint['model'])
            model = model.eval().cuda()
            print(f'Loaded model from checkpoint: {checkpoint_path}')
            print(f'Epoch: {checkpoint["epoch"]}, Best Acc: {checkpoint["accuracy"]}')
            return model, transform
        else:
            model = timm.create_model(f'hf-hub:timm/{model_name}', pretrained=True).eval().cuda()
            data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
            transform = timm.data.create_transform(**data_cfg) 
    else:
        model = timm.create_model(model_name, pretrained=True).eval().cuda()
        data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
        transform = timm.data.create_transform(**data_cfg)
    return model, transform


def to_human_format_str(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])
