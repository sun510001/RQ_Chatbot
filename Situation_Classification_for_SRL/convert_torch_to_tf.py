# -*- coding:utf-8 -*-

from transformers import TFRobertaModel, RobertaConfig, RobertaTokenizer, BertConfig, TFBertModel, BertTokenizer


class Convert_Model(object):
    @staticmethod
    def dl_roberta(model_name, path):
        print("Start to download", model_name, "...")
        dump = path + model_name
        config = RobertaConfig.from_pretrained(model_name)
        model = TFRobertaModel.from_pretrained(model_name)
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        config.save_pretrained(dump)
        model.save_pretrained(dump)
        tokenizer.save_pretrained(dump)
        print("Download", model_name, "completed.")


    @staticmethod
    def dl_bert(model_name, path):
        print("Start to download", model_name, "...")
        dump = path + model_name
        config = BertConfig.from_pretrained(model_name)
        model = TFBertModel.from_pretrained(model_name)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        config.save_pretrained(dump)
        model.save_pretrained(dump)
        tokenizer.save_pretrained(dump)
        print("Download", model_name, "completed.")
