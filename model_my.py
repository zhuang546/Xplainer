from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

# —— 正确的导入 —— #
from health_multimodal.image import get_image_inference
from health_multimodal.image.utils import ImageModelType
from health_multimodal.text import get_bert_inference
from health_multimodal.text.utils import BertEncoderType
from health_multimodal.vlp import ImageTextInferenceEngine

from utils import cos_sim_to_prob, prob_to_log_prob, log_prob_to_prob

class InferenceModel:
    def __init__(self):
        # 文本推理（CXR-BERT）
        self.text_inference = get_bert_inference(
            bert_encoder_type=BertEncoderType.CXR_BERT # 老模型 CXR_BERT, 新模型 BIOVIL_T_BERT
        )

        # 图像推理（BioViL-ResNet）
        self.image_inference = get_image_inference(
            image_model_type=ImageModelType.BIOVIL # 老模型 BIOVIL，新模型 BIOVIL_T
        )

        # 跨模态推理器
        self.image_text_inference = ImageTextInferenceEngine(
            image_inference_engine=self.image_inference,
            text_inference_engine=self.text_inference,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_text_inference.to(self.device)

        # 简单缓存
        self.text_embedding_cache = {}
        self.image_embedding_cache = {}

        # 与模型匹配的 Transform
        self.transform = self.image_inference.transform
    
    def get_similarity_score_from_raw_data(self, image_embedding, query_text: str) -> float:
        """Compute the cosine similarity score between an image and one or more strings.
        If multiple strings are passed, their embeddings are averaged before L2-normalization.
        :param image_path: Path to the input chest X-ray, either a DICOM or JPEG file.
        :param query_text: Input radiology text phrase.
        :return: The similarity score between the image and the text.
        """
        assert not self.image_text_inference.image_inference_engine.model.training
        assert not self.image_text_inference.text_inference_engine.model.training
        if query_text in self.text_embedding_cache:
            text_embedding = self.text_embedding_cache[query_text]
        else:
            text_embedding = self.image_text_inference.text_inference_engine.get_embeddings_from_prompt([query_text], normalize=False)
            text_embedding = text_embedding.mean(dim=0)
            text_embedding = F.normalize(text_embedding, dim=0, p=2)
            self.text_embedding_cache[query_text] = text_embedding

        cos_similarity = image_embedding @ text_embedding.t()

        return cos_similarity.item()

    def process_image(self, image):
        ''' same code as in image_text_inference.image_inference_engine.get_projected_global_embedding() but adapted to deal with image instances instead of path'''

        transformed_image = self.transform(image)
        projected_img_emb = self.image_inference.model.forward(transformed_image).projected_global_embedding
        projected_img_emb = F.normalize(projected_img_emb, dim=-1)
        assert projected_img_emb.shape[0] == 1
        assert projected_img_emb.ndim == 2
        return projected_img_emb[0]

    def get_descriptor_probs(self, image_path: Path, disease_descriptors: dict, do_negative_prompting=True, demo=False):
        probs = {}
        negative_probs = {}
        if image_path in self.image_embedding_cache:
            image_embedding = self.image_embedding_cache[image_path]
        else:
            image_embedding = self.image_text_inference.image_inference_engine.get_projected_global_embedding(image_path)
            if not demo:
                self.image_embedding_cache[image_path] = image_embedding

        # Default get_similarity_score_from_raw_data would load the image every time. Instead we only load once.
        for disease, descs in disease_descriptors.items():
            for desc in descs:
                prompt_key = f"{desc} indicating {disease}"
                prompt = f"There is {desc} indicating {disease}"
                score = self.get_similarity_score_from_raw_data(image_embedding, prompt)
                #print(f"Prompt:\n{prompt}\n, Score: {score:.4f}")
                if do_negative_prompting:
                    neg_prompt = f"There is no {desc}, no {disease}"
                    neg_score = self.get_similarity_score_from_raw_data(image_embedding, neg_prompt)

                pos_prob = cos_sim_to_prob(score) # 在不进行负面提示的情况下，直接使用余弦相似度转换为概率

                if do_negative_prompting: # 如果进行负面提示，则计算一个统合概率来覆盖掉正向概率
                    pos_prob, neg_prob = torch.softmax((torch.tensor([score, neg_score]) / 0.5), dim=0)
                    negative_probs[prompt_key] = neg_prob

                probs[prompt_key] = pos_prob

        return probs, negative_probs

    def get_all_descriptors_only_disease(self, disease_descriptors):
        all_descriptors = set()
        for disease, descs in disease_descriptors.items():
            all_descriptors.update([f"{desc}" for desc in descs])
        all_descriptors = sorted(all_descriptors)
        return all_descriptors

    def get_diseases_probs(self, disease_descriptors, pos_probs, negative_probs, prior_probs=None, do_negative_prompting=True):
        disease_probs = {}
        disease_neg_probs = {}

        # Define temperature for LogSumExp aggregation for each disease.
        # A small temp (e.g., 0.05) makes it behave like 'max'.
        # 'mean' will be used for diseases not in this dict.
        temp_strategies = {
            'No Finding': None,
            'Enlarged Cardiomediastinum': None,
            'Cardiomegaly': None,
            'Lung Opacity': None,
            'Lung Lesion': None,
            'Edema': None,
            'Consolidation': None,
            'Pneumonia': None,
            'Atelectasis': None,
            'Pneumothorax': None,
            'Pleural Effusion': None,
            'Pleural Other': 0.05,
            'Fracture': 0.05,
            'Support Devices': 0.01,
        }

        for disease, descriptors in disease_descriptors.items():
            desc_log_probs = []
            desc_neg_log_probs = []
            for desc in descriptors:
                desc = f"{desc} indicating {disease}"
                desc_log_probs.append(prob_to_log_prob(pos_probs[desc]))
                if do_negative_prompting:
                    desc_neg_log_probs.append(prob_to_log_prob(negative_probs[desc]))

            # Apply strategy based on the disease
            temp = temp_strategies.get(disease)
            #temp = 0.05
            temp = None  # Default to mean for all other diseases

            if temp is not None:
                # Use LogSumExp for specified diseases
                log_probs_tensor = torch.tensor(desc_log_probs)
                disease_log_prob = (torch.logsumexp(log_probs_tensor/temp, dim=0) - torch.log(torch.tensor(len(desc_log_probs)))) * temp
                if do_negative_prompting:
                    # For negative prompts, we can use a similar logic or a different one.
                    log_neg_probs_tensor = torch.tensor(desc_neg_log_probs)
                    disease_neg_log_prob = (torch.logsumexp(log_neg_probs_tensor/temp, dim=0) - torch.log(torch.tensor(len(desc_neg_log_probs)))) * temp
            else:
                # Default to mean for all other diseases
                disease_log_prob = sum(sorted(desc_log_probs, reverse=True)) / len(desc_log_probs)
                if do_negative_prompting:
                    disease_neg_log_prob = sum(desc_neg_log_probs) / len(desc_neg_log_probs)

            disease_probs[disease] = log_prob_to_prob(disease_log_prob)
            if do_negative_prompting:
                disease_neg_probs[disease] = log_prob_to_prob(disease_neg_log_prob)

        return disease_probs, disease_neg_probs


    # Threshold Based
    def get_predictions(self, disease_descriptors, threshold, disease_probs, keys):
        predicted_diseases = []
        prob_vector = torch.zeros(len(keys), dtype=torch.float)  # num of diseases
        for idx, disease in enumerate(disease_descriptors):
            if disease == 'No Finding':
                continue
            prob_vector[keys.index(disease)] = disease_probs[disease]
            if disease_probs[disease] > threshold:
                predicted_diseases.append(disease)

        # prob_vector[0] = 1.0 - max(prob_vector) # No finding rule based
        # A smooth approximation of 1 - max(p) using LogSumExp
        # This captures the "probability of any disease being present" in a soft way.
        disease_probs_tensor = prob_vector[1:]
        # Add a small epsilon to prevent log(0)
        epsilon = 1e-6
        logits = torch.log(disease_probs_tensor + epsilon) - torch.log(1 - disease_probs_tensor + epsilon)
        
        # Aggregate logits using LogSumExp, which is a smooth max function
        # A temperature parameter could be added here for more control: torch.logsumexp(logits / temp, 0) * temp
        prob_any_disease = torch.sigmoid(torch.logsumexp(logits, 0))
        prob_vector[0] = 1.0 - prob_any_disease

        return predicted_diseases, prob_vector

    # Negative vs Positive Prompting
    def get_predictions_bin_prompting(self, disease_descriptors, disease_probs, negative_disease_probs, keys):
        predicted_diseases = []
        prob_vector = torch.zeros(len(keys), dtype=torch.float)  # num of diseases
        for idx, disease in enumerate(disease_descriptors):
            if disease == 'No Finding':
                continue
            pos_neg_scores = torch.tensor([disease_probs[disease], negative_disease_probs[disease]])
            prob_vector[keys.index(disease)] = pos_neg_scores[0]
            if torch.argmax(pos_neg_scores) == 0:  # Positive is More likely
                predicted_diseases.append(disease)

        # prob_vector[0] = torch.prod(1.0 - prob_vector[1:])
        # A smooth approximation of 1 - max(p) using LogSumExp
        disease_probs_tensor = prob_vector[1:]
        epsilon = 1e-6
        logits = torch.log(disease_probs_tensor + epsilon) - torch.log(1 - disease_probs_tensor + epsilon)
        # A temperature parameter could be added here for more control
        temp = 0.05  # Lower temperature makes it closer to a hard max
        prob_any_disease = torch.sigmoid(torch.logsumexp(logits / temp, 0) * temp)
        prob_vector[0] = 1.0 - prob_any_disease

        return predicted_diseases, prob_vector
