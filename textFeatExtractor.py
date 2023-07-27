import os
import numpy as np
import paddle
from psutil import cpu_count
from paddlenlp.transformers import AutoTokenizer,BertTokenizer
import onnxruntime
MODEL_HOME = os.path.dirname(__file__)+"/models"

class InferBackend(object):

    def __init__(self,
                 model_path,
                 device='cpu',
                 cpu_backend="mkldnn",
                 precision_mode="fp32",
                 infer_type = "onnx",
                 num_threads=10):
        """
        Args:
            model_path (str): The model path for deployment.
            device (str): The deployed device can be set to cpu, gpu or xpu, the default is cpu.
            cpu_backend (str): Inference backend when deploy on cpu, which can be mkldnn or onnxruntime, 
                                the default is mkldnn.
            precision_mode (str): Inference precision, which can be fp32, fp16 or int8, the default is fp32.
            infer_type (str): model infer backend, 'Inference' or 'onnx'
            num_threads (int): Number of cpu threads during inference, the default is 10.
        """
        self.predictor_type = infer_type
        precision_mode = precision_mode.lower()
        model_path = self.model_path_correction(model_path)
        # Check if the model is a quantized model
        # is_int8_model = self.paddle_quantize_model(model_path)
        print(">>> [InferBackend] Creating Engine ...")

        if self.predictor_type == "Inference":
            from paddle import inference

            config = paddle.inference.Config(model_path + ".pdmodel",
                                             model_path + ".pdiparams")
            # quantized model on GPU
            if device == 'gpu':
                config.enable_use_gpu(100, 0)
                precision_type = inference.PrecisionType.Float32
            elif device == 'cpu':
                config.disable_gpu()
                config.switch_ir_optim(True)
                if cpu_backend == "mkldnn":
                    config.enable_mkldnn()
                elif cpu_backend == "onnxruntime":
                    config.enable_onnxruntime()
                    config.enable_ort_optimization()
            config.enable_memory_optim()
            config.set_cpu_math_library_num_threads(num_threads)
            config.delete_pass("embedding_eltwise_layernorm_fuse_pass")
            self.predictor = paddle.inference.create_predictor(config)
            self.input_names = [
                name for name in self.predictor.get_input_names()
            ]
            self.input_handles = [
                self.predictor.get_input_handle(name)
                for name in self.predictor.get_input_names()
            ]
            self.output_handles = [
                self.predictor.get_output_handle(name)
                for name in self.predictor.get_output_names()
            ]
        elif self.predictor_type == "onnx":
            try:
                so = onnxruntime.SessionOptions()
                so.log_severity_level = 3
                self.session = onnxruntime.InferenceSession(f"{model_path}.onnx",so)
                self.session.get_modelmeta()
                self.input_handles = [a.name for a in self.session.get_inputs()]
                self.output_handles =  [a.name for a in self.session.get_outputs()]
            except Exception as ex:
                print(ex)
                self.session = None
        print(">>> [InferBackend] Engine Created ...")

    def model_path_correction(self, model_path):
        if self.predictor_type == "Inference":
            if os.path.isfile(model_path + ".pdmodel"):
                return model_path
            new_model_path = None
            for file in os.listdir(model_path):
                if (file.count(".pdmodel")):
                    filename = file[:-8]
                    new_model_path = os.path.join(model_path, filename)
                    return new_model_path
        elif self.predictor_type == 'onnx':
            if os.path.isfile(model_path + ".onnx"):
                return model_path
            new_model_path = None
            for file in os.listdir(model_path):
                if (file.count(".onnx")):
                    filename = file[:-5]
                    new_model_path = os.path.join(model_path, filename)
                    return new_model_path
            
        assert new_model_path is not None, "Can not find model file in your path."


    def infer(self, input_dict: dict):
        if self.predictor_type == "Inference":
            for idx, input_name in enumerate(self.input_names):
                self.input_handles[idx].copy_from_cpu(input_dict[input_name])
            self.predictor.run()
            output = [
                output_handle.copy_to_cpu()
                for output_handle in self.output_handles
            ]
            return output
        elif self.predictor_type == "onnx":
            net_inputs = {k: input_dict[k] for k in self.input_handles}
            result = self.session.run(None, net_inputs)
        return result


class TextFeatureExtractor(object):

    def __init__(self, device = "cpu",
                task_name = "simbert-base-chinese",    # 
                max_seq_length = 128):
        if not isinstance(device, str):
            print(
                ">>> [InferBackend] The type of device must be string, but the type you set is: ",
                type(device))
            exit(0)
        device = device.lower()
        if device not in ['cpu', 'gpu', 'xpu']:
            print(
                ">>> [InferBackend] The device must be cpu or gpu, but your device is set to:",
                type(device))
            exit(0)

        self.supported_models = ['simbert-base-chinese','simbert-small-chinese','simbert-tiny-chinese']
        self.task_name = task_name
        self.model_name = task_name.replace("-","_")

        self.model_name_or_path = os.path.join(MODEL_HOME,self.model_name)
        self.model_path = self.model_name_or_path+"/"+self.model_name
        self.config_path = self.model_name_or_path+"/tokenizer_config.json"
        self.vocab_file = self.model_name_or_path+"/vocab.txt"
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path,
                                                        use_fast=False)
        if self.tokenizer is None:
            # self.tokenizer = AutoTokenizer._get_tokenizer_class_from_config(self.model_name_or_path,
            #                                             use_fast=False)
            self.tokenizer = BertTokenizer(self.vocab_file)
        
        if task_name in self.supported_models:
            self.preprocess = self.seq_feat_preprocess
            self.postprocess = self.seq_feat_postprocess
        else:
            print(
                f"{self.supported_models}"
            )
            exit(0)

        self.max_seq_length = max_seq_length

        if device == 'cpu':
            self.set_dynamic_shape = False
            self.shape_info_file = None
            self.batch_size =32
            self.num_threads = cpu_count(logical=False)
            
        if device == 'gpu':
            self.num_threads = cpu_count(logical=False)
            self.batch_size = 32
            self.shape_info_file = None
            self.set_dynamic_shape = False

        self.inference_backend = InferBackend(
            self.model_path,
            device=device,
            # infer_type="Inference",
            num_threads=self.num_threads)

    def seq_feat_preprocess(self, input_data: list):
        '''
        文本特征提取预处理
        '''
        data = input_data
        # tokenizer + pad
        data = self.tokenizer(data,
                              max_length=self.max_seq_length,
                              padding=True,
                              truncation=True,
                              )
        
        input_ids = data["input_ids"]
        token_type_ids = data["token_type_ids"]
        return {
            "input_ids": np.array(input_ids, dtype="int64"),
            "token_type_ids": np.array(token_type_ids, dtype="int64")
        }


    def seq_feat_postprocess(self, infer_data):
        '''
        文本特征提取后处理
        '''
        vecs_text = infer_data[1]
        # vecs_text = vecs_text / (vecs_text**2).sum(axis=1, keepdims=True)**0.5
        return vecs_text

    def infer(self, data):
        return self.inference_backend.infer(data)

    def predict(self, input_data: list):
        preprocess_result = self.preprocess(input_data)
        infer_result = self.infer(preprocess_result)
        result = self.postprocess(infer_result)
        return result
