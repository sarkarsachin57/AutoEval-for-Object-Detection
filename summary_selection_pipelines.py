from init import *


class Img2Vec():
    RESNET_OUTPUT_SIZES = {
        'resnet18': 512,
        'resnet34': 512,
        'resnet50': 2048,
        'resnet101': 2048,
        'resnet152': 2048,
    }

    def __init__(self, cuda=False, model='resnet-18', layer='default', layer_output_size=512):
        """ Img2Vec
        :param cuda: If set to True, will run forward pass on GPU
        :param model: String name of requested model
        :param layer: String or Int depending on model.
        :param layer_output_size: Int depicting the output size of the requested layer
        """
        print("Using: ", model, " for feature extraction.")
        self.device = torch.device("cuda" if cuda else "cpu")

        self.model_name = model
        self.layer_output_size = layer_output_size

        self.model, self.extraction_layer = self._get_model_and_layer(
            model, layer)

        self.model = self.model.to(self.device)

        self.model.eval()

        self.scaler = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def get_vec(self, img):
        """ Get vector embedding from PIL image
        :param img: PIL Image
        :param tensor: If True, get_vec will return a FloatTensor instead of Numpy array
        :returns: Numpy ndarray
        """
        if type(img) == list:
            a = [self.normalize(self.to_tensor(self.scaler(im))) for im in img]
            images = torch.stack(a).to(self.device)
            if self.model_name in ['alexnet', 'vgg']:
                my_embedding = torch.zeros(len(img), self.layer_output_size)
            elif self.model_name == 'densenet':
                my_embedding = torch.zeros(len(img), self.layer_output_size, 7, 7)
            else:
                my_embedding = torch.zeros(len(img), self.layer_output_size, 1, 1)

            def copy_data(m, i, o):
                my_embedding.copy_(o.data)

            h = self.extraction_layer.register_forward_hook(copy_data)
            with torch.no_grad():
                h_x = self.model(images)
            h.remove()

            if self.model_name in ['alexnet', 'vgg']:
                return my_embedding.numpy()[:, :]
            elif self.model_name == 'densenet':
                return torch.mean(my_embedding, (2, 3), True).numpy()[:, :, 0, 0]
            else:
                return my_embedding.numpy()[:, :, 0, 0]
        else:
          image = self.normalize(self.to_tensor(
              self.scaler(img))).unsqueeze(0).to(self.device)

          if self.model_name in ['alexnet', 'vgg']:
              my_embedding = torch.zeros(1, self.layer_output_size)
          elif self.model_name == 'densenet':
              my_embedding = torch.zeros(1, self.layer_output_size, 7, 7)
          else:
              my_embedding = torch.zeros(1, self.layer_output_size, 1, 1)

          def copy_data(m, i, o):
              my_embedding.copy_(o.data)

          h = self.extraction_layer.register_forward_hook(copy_data)
          with torch.no_grad():
              h_x = self.model(image)
          h.remove()

          if self.model_name in ['alexnet', 'vgg']:
              return my_embedding.numpy()[0, :]
          elif self.model_name == 'densenet':
              return torch.mean(my_embedding, (2, 3), True).numpy()[0, :, 0, 0]
          else:
              return my_embedding.numpy()[0, :, 0, 0]

    def _get_model_and_layer(self, model_name, layer):
        """ Internal method for getting layer from model
        :param model_name: model name such as 'resnet-18'
        :param layer: layer as a string for resnet-18 or int for alexnet
        :returns: pytorch model, selected layer
        """
        if model_name.startswith('resnet') and not model_name.startswith('resnet-'):
            model = getattr(models, model_name)(pretrained=True)
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = self.RESNET_OUTPUT_SIZES[model_name]
            else:
                layer = model._modules.get(layer)
            return model, layer
        elif model_name == 'resnet-18':
            model = models.resnet18(pretrained=True)
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = 512
            else:
                layer = model._modules.get(layer)

            return model, layer

        elif model_name == 'alexnet':
            model = models.alexnet(pretrained=True)
            if layer == 'default':
                layer = model.classifier[-2]
                self.layer_output_size = 4096
            else:
                layer = model.classifier[-layer]

            return model, layer

        elif model_name == 'vgg':
            # VGG-11
            model = models.vgg11_bn(pretrained=True)
            if layer == 'default':
                layer = model.classifier[-2]
                self.layer_output_size = model.classifier[-1].in_features # should be 4096
            else:
                layer = model.classifier[-layer]

            return model, layer

        elif model_name == 'densenet':
            # Densenet-121
            model = models.densenet121(pretrained=True)
            if layer == 'default':
                layer = model.features[-1]
                self.layer_output_size = model.classifier.in_features # should be 1024
            else:
                raise KeyError('Un support %s for layer parameters' % model_name)

            return model, layer
        elif model_name == 'googlenet':
            model = models.googlenet(pretrained=True)
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = 1024
            else:
                layer = model._modules.get(layer)

            return model, layer
        else:
            raise KeyError('Model %s was not found' % model_name)


img2vec = Img2Vec(cuda=True if torch.cuda.is_available() else False, model='resnet50')

def get_diverse_set_of_images(image_paths, n_select):

    print('-- Applying Diversity Selection --')
    
    print('Loading images...')
    imgs = []
    image_list = []
    index = 1
    for image_path in tqdm(image_paths):
        img = Image.open(image_path).convert('RGB')
        imgs.append(img)
        image_list.append(image_path)
        index += 1
    
    print('Extracting features from images...')
    vectors = img2vec.get_vec(imgs)
    K_dense = create_kernel(vectors, 'cosine')
    obj = FacilityLocationFunction(n=K_dense.shape[0], mode="dense", sijs=K_dense, separate_rep=False,pybind_mode="array")
    selected_idx = obj.maximize(budget=n_select,optimizer='NaiveGreedy')

    return [idx for idx, _ in selected_idx]


def get_summary_images(imagewise_stats, acc_filter_thresh, prec_filter_thresh, recl_filter_thresh, fp_filter_thresh, fn_filter_thresh, false_filter_thresh, low_conf_filter_thresh, n_overlap_thresh, n_select):

    filtered = np.logical_and(np.array(imagewise_stats['ACC']) <= acc_filter_thresh, np.array(imagewise_stats['PREC']) <= prec_filter_thresh)
    filtered = np.logical_and(filtered, np.array(imagewise_stats['RECL']) <= recl_filter_thresh)
    filtered = np.logical_and(filtered, np.array(imagewise_stats['FP']) >= fp_filter_thresh)
    filtered = np.logical_and(filtered, np.array(imagewise_stats['FN']) >= fn_filter_thresh)
    filtered = np.logical_and(filtered, np.array(imagewise_stats['FP']) + np.array(imagewise_stats['FN']) >= false_filter_thresh)
    filtered = np.logical_and(filtered, np.array(imagewise_stats['nLowConf']) >= low_conf_filter_thresh)
    final_filtered_list = np.where(np.logical_and(filtered, np.array(imagewise_stats['nOverLap']) >= n_overlap_thresh))[0]
    final_filtered_raw_frames = np.array(imagewise_stats['raw_save_path'])[final_filtered_list].tolist()
    final_filtered_tps = np.array(imagewise_stats['TP'])[final_filtered_list].tolist()
    final_filtered_fps = np.array(imagewise_stats['FP'])[final_filtered_list].tolist()
    final_filtered_fns = np.array(imagewise_stats['FN'])[final_filtered_list].tolist()
    
    final_summary_lists = {}

    if len(final_filtered_list) > n_select:
        
        final_filtered_list = get_diverse_set_of_images(final_filtered_raw_frames, n_select)
                
        final_summary_lists['raw_save_path'] = np.array(imagewise_stats['raw_save_path'])[final_filtered_list].tolist()
        final_summary_lists['dets_save_path'] = np.array(imagewise_stats['dets_save_path'])[final_filtered_list].tolist()
        final_summary_lists['aux_dets_save_path'] = np.array(imagewise_stats['aux_dets_save_path'])[final_filtered_list].tolist()
        final_summary_lists['tp_fp_fn_save_path'] = np.array(imagewise_stats['tp_fp_fn_save_path'])[final_filtered_list].tolist()
        final_summary_lists['low_conf_save_path'] = np.array(imagewise_stats['low_conf_save_path'])[final_filtered_list].tolist()
        final_summary_lists['overlap_save_path'] = np.array(imagewise_stats['overlap_save_path'])[final_filtered_list].tolist()

        # final_filtered_list = random.sample([i for i in range(len(final_filtered_list))], n_select)

        # final_filtered_raw_frames = np.array(imagewise_stats['raw_save_path'])[final_filtered_list].tolist()

        final_filtered_tps = np.array(imagewise_stats['TP'])[final_filtered_list].tolist()
        final_filtered_fps = np.array(imagewise_stats['FP'])[final_filtered_list].tolist()
        final_filtered_fns = np.array(imagewise_stats['FN'])[final_filtered_list].tolist()
        

        acc = 100*(2*sum(final_filtered_tps) + 1e-10) / (2*sum(final_filtered_tps) + sum(final_filtered_fps) + sum(final_filtered_fns) + 1e-10)
        prec = 100*(sum(final_filtered_tps) + 1e-10) / (sum(final_filtered_tps) + sum(final_filtered_fps) + 1e-10)
        recl = 100*(sum(final_filtered_tps) + 1e-10) / (sum(final_filtered_tps) + sum(final_filtered_fns) + 1e-10)

        # final_summary_lists['raw_save_path'] = final_filtered_raw_frames
        final_summary_lists['ACC'] = round(acc, 2)
        final_summary_lists['PREC'] = round(prec, 2)
        final_summary_lists['RECL'] = round(recl, 2)

        print('Summary Length :', len(final_filtered_list))
        print(f'Estimated Accuracy : {round(acc, 2)}, Precision : {round(prec, 2)}, Recall : {round(recl, 2)}')


        return final_summary_lists

    acc = 100*(2*sum(final_filtered_tps) + 1e-10) / (2*sum(final_filtered_tps) + sum(final_filtered_fps) + sum(final_filtered_fns) + 1e-10)
    prec = 100*(sum(final_filtered_tps) + 1e-10) / (sum(final_filtered_tps) + sum(final_filtered_fps) + 1e-10)
    recl = 100*(sum(final_filtered_tps) + 1e-10) / (sum(final_filtered_tps) + sum(final_filtered_fns) + 1e-10)

    print('Summary Length :', len(final_filtered_list))
    print(f'Estimated Accuracy : {round(acc, 2)}, Precision : {round(prec, 2)}, Recall : {round(recl, 2)}')


    final_summary_lists['raw_save_path'] = final_filtered_raw_frames
    final_summary_lists['dets_save_path'] = np.array(imagewise_stats['dets_save_path'])[final_filtered_list].tolist()
    final_summary_lists['aux_dets_save_path'] = np.array(imagewise_stats['aux_dets_save_path'])[final_filtered_list].tolist()
    final_summary_lists['tp_fp_fn_save_path'] = np.array(imagewise_stats['tp_fp_fn_save_path'])[final_filtered_list].tolist()
    final_summary_lists['low_conf_save_path'] = np.array(imagewise_stats['low_conf_save_path'])[final_filtered_list].tolist()
    final_summary_lists['overlap_save_path'] = np.array(imagewise_stats['overlap_save_path'])[final_filtered_list].tolist()

    final_summary_lists['ACC'] = round(acc, 2)
    final_summary_lists['PREC'] = round(prec, 2)
    final_summary_lists['RECL'] = round(recl, 2)


    return final_summary_lists


# def get_overall_summary_from_all_images_and_videos(multi_stats_json, 
#                                                    acc_filter_thresh=50, 
#                                                    prec_filter_thresh=50, 
#                                                    recl_filter_thresh=70, 
#                                                    low_conf_filter_thresh=50, 
#                                                    n_overlap_thresh=2, 
#                                                    n_select=30):

#     imagewise_stats = multi_stats_json['imagewise_stats'][-1]
#     return get_summary_images(imagewise_stats, 
#                        acc_filter_thresh=acc_filter_thresh, 
#                        prec_filter_thresh=prec_filter_thresh, 
#                        recl_filter_thresh=recl_filter_thresh, 
#                        low_conf_filter_thresh=low_conf_filter_thresh, 
#                        n_overlap_thresh=2, 
#                        n_select=n_select)


def get_overall_summary_from_all_images_and_videos(multi_stats_json, 
                                                   acc_filter_thresh, 
                                                   prec_filter_thresh, 
                                                   recl_filter_thresh, 
                                                   fp_filter_thresh, 
                                                   fn_filter_thresh, 
                                                   false_filter_thresh, 
                                                   low_conf_filter_thresh, 
                                                   n_overlap_thresh,
                                                   n_select):

    imagewise_stats = multi_stats_json['imagewise_stats'][-1]
    return get_summary_images(imagewise_stats, 
                        acc_filter_thresh, 
                        prec_filter_thresh, 
                        recl_filter_thresh, 
                        fp_filter_thresh, 
                        fn_filter_thresh, 
                        false_filter_thresh, 
                        low_conf_filter_thresh, 
                        n_overlap_thresh,
                        n_select=n_select)