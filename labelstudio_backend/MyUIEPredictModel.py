import numpy as np
import os
from paddlenlp import Taskflow
import json
from label_studio_ml.model import LabelStudioMLBase
from torch.cuda import empty_cache
import gc
import pandas as pd
#Set up label format
PERSON = '原告'
CHARGE_MONEY = '非財產上之損害的求償精神慰撫金'
JUDGE_MONEY = '法院得心證判斷之適當裁定精神慰撫金'
OTHER_MONEY = '其他精神慰撫金'
PARSED_LABEL_CONFIG = {'label': {
    'type': 'Labels',
    'to_name': ['text'],
    'inputs': [{
        'type': 'Text',
        'value': 'text'
    }],
    'labels': [PERSON, CHARGE_MONEY, JUDGE_MONEY, OTHER_MONEY],
    'labels_attrs': {
        PERSON: {
            'value': PERSON,
            'background': '#44ff00'
        },
        CHARGE_MONEY: {
            'value': CHARGE_MONEY,
            'background': '#0062ff'
        },
        JUDGE_MONEY: {
            'value': JUDGE_MONEY,
            'background': '#9e06e5'
        },
        OTHER_MONEY: {
            'value': OTHER_MONEY,
            'background': '#ff0000'
        }
    }
}
}
LABEL_SCHEMA = [CHARGE_MONEY, JUDGE_MONEY,
                {PERSON: CHARGE_MONEY,
                 PERSON: JUDGE_MONEY}]

#Set up inference model
PRECISION = 'fp32' #GPU for fp16
MODEL_PATH = "./NLLP/experiment/PaddleNLP_UIE/data_v3/checkpoint/model_best"

#Set up label schema on label-studio
DEL_OVERLAP_ENTITY = True
DEL_REPEAT_PERSON = True
ADD_RELATION = True

#Set up learning loop parameter
START_LEARNING_LOOP = True
RETRAINING_DATA_SIZE = 10
STORE_TRAINING_DATA_HISTORY = True
STORE_PATH = "./NLLP/experiment/labelstudio"
LEARNING_FILE_PATH = STORE_PATH + "/ml_backend_tmp_decanno.jsonl"

#Set up model training parameter
BATCH_SIZE = 16

class MyUIEPredictModel(LabelStudioMLBase):
    def __init__(self, **kwargs) -> None:
        print('======== in init ========')
        super(MyUIEPredictModel, self).__init__(**kwargs)
        self.parsed_label_config = PARSED_LABEL_CONFIG
        self.from_name, self.info = list(self.parsed_label_config.items())[0]
        self.to_name = self.info['to_name'][0]
        self.value = self.info['inputs'][0]['value']
        self.labels = list(self.info['labels'])

        self.dynamic_param = dict(pd.read_json("./my_ml_backend/Dynamic_Config.json", typ='series'))
        #self.accumulate_task_counter = 1
        #self.total_entity_id = 0
        #self.decanno_ext_id = 1
        #self.data_docano_format = []

        #print(os.listdir(MODEL_PATH))
        self.model = Taskflow("information_extraction", schema=LABEL_SCHEMA,
                                task_path=MODEL_PATH,
                                precision=PRECISION)


    def _del_overlap_entity(self, result):
        index = 0
        while index < len(result) - 1:
            if result[index]['value']['start'] == result[index + 1]['value']['start'] :
                if result[index]['value']['score'] > result[index + 1]['value']['score']:
                    result.pop(index + 1)
                else:
                    result.pop(index)
            else:
                index += 1
        return result


    def _add_relation(self, result, relations):
        start_id_mapping = {result[index]['value']['start']: result[index]['id'] for index in range(len(result))}

        for relation in relations:
            people_start = relation['start']
            for relation_type in relation['relations']:
                money_start = relation['relations'][relation_type][0]['start']
                if start_id_mapping.get(people_start) and start_id_mapping.get(money_start):
                    result.append({
                        'from_id': start_id_mapping[people_start],
                        'to_id': start_id_mapping[money_start],
                        "type": "relation",
                        "direction": "right",
                        "labels": [
                            relation_type
                        ]
                    })
        return result
    

    def _del_repeat_person(self, result):
        #This function must be executed after sorted
        unique_name_map = []
        result_index = []
        for i, item in enumerate(result):
            print(item)
            if item['value']['labels'][0] == PERSON:
                if item['value']['text'] not in unique_name_map:
                    unique_name_map.append(item['value']['text'])
                    result_index.append(i)
            else:
                result_index.append(i)
        return list(np.array(result)[result_index])

    
    def _labelstudio2docano(self, labelstudio_output):
        id = self.dynamic_param['decanno_ext_id']
        text = labelstudio_output['data']['task']['data']['text']
        entities = []
        relations = []

        for label_result in labelstudio_output['data']['annotation']['result']:
            #entity
            if label_result.get('value') is not None:
                entities.append({
                    "id": label_result['id'],
                    "label": label_result['value']['labels'][0],
                    "start_offset": label_result['value']['start'],
                    "end_offset": label_result['value']['end']
                })
            else: #relation
                relations.append({
                    "id": label_result['from_id'] + label_result['to_id'],
                    "from_id": label_result['from_id'],
                    "to_id": label_result['to_id'],
                    "type": label_result['labels'][0]
                })
                
        self.dynamic_param['decanno_ext_id'] += 1
        return {"id": id, "text": text, "entities": entities, "relations": relations}


    def write_dynamic_records(self, json_file="./my_ml_backend/Dynamic_Config.json"):
        with open(json_file, 'w') as f:
            json.dump(self.dynamic_param, f, ensure_ascii=False)


    def predict(self, tasks, **kwargs):
        print('======== in predict ========')
        from_name = self.from_name
        to_name = self.to_name
        model = self.model
        predictions = []
        #print("tasks: ", tasks)
        for task in tasks:
            print("------------------------------------")
            print("Predict task number: ", self.dynamic_param['accumulate_task_counter'])
            #print("Predict task:", task)
            text = task['data'][self.value]
            uie = model(text)[0]
            #print(uie)
            result = []
            scores = []
            relations = []

            for key in uie:
                #print("key: ", key)
                for item in uie[key]:
                    #print('item: ', item)
                    result.append({
                        'value': {
                            'start': item['start'],
                            'end': item['end'],
                            'score': item['probability'],
                            'text': item['text'],
                            'labels': [key]
                        },
                        'from_name': from_name, 
                        'to_name': to_name,
                        'type': 'labels',
                        'id': str(self.dynamic_param['total_entity_id'])
                    })
                    self.dynamic_param['total_entity_id'] += 1
                    if item.get('relations') is not None:
                        relations.append(item)
                    scores.append(item['probability'])
                        
            result = sorted(result, key=lambda x: x['value']['start'])
            print("========== result before ==========")
            mean_score = np.mean(scores) if len(scores) else 0

            #format for "extract_spiritual_money" task
            #delete overlap entity
            if DEL_OVERLAP_ENTITY:
                result = self._del_overlap_entity(result)
            #delete repeat person
            if DEL_REPEAT_PERSON:
                result = self._del_repeat_person(result)
            #add relation
            if ADD_RELATION:
                result = self._add_relation(result, relations)
                

            predictions.append({
                'result': result,
                'score': float(mean_score),
                'model_version': 'uie_new'
            })

            print("========== result after ==========")
            print("result: ", result)
            self.dynamic_param['accumulate_task_counter'] += 1
            print()
        
        self.write_dynamic_records()
        return predictions


    def fit(self, annotation, workdir=None, **kwargs):
        #train submit or update會跑到這邊
        print('======== in fit ========')
        print(len(self.dynamic_param['data_docano_format']))
        if START_LEARNING_LOOP:
            '''
            print(kwargs['data'].keys())
            print("kwargs.data.annotation: ", kwargs['data']['annotation'])
            print()
            print("kwargs.data.project: ", kwargs['data']['project'])
            print()
            print("kwargs.data.task: ", kwargs['data']['task'])
            '''
            #這邊把label-studio output轉成decanno格式
            #a = dict(json.dumps(self._labelstudio2docano(kwargs), ensure_ascii=False))
            #print(a)
            #self.dynamic_param['data_docano_format'].append(json.dumps(self._labelstudio2docano(kwargs), ensure_ascii=False))
            self.dynamic_param['data_docano_format'].append(self._labelstudio2docano(kwargs))
            
            if len(self.dynamic_param['data_docano_format']) == RETRAINING_DATA_SIZE:
                with open(LEARNING_FILE_PATH, 'a', encoding='utf8') as f:
                    for i in self.dynamic_param['data_docano_format']:
                        json.dump(i, f, ensure_ascii=False)
                        f.write("\n")
                        
                
                #store training data
                if STORE_TRAINING_DATA_HISTORY:
                    path = STORE_PATH + "/ml_backend_history.txt"
                    with open(path, 'a', encoding='utf8') as f:
                        for i in self.dynamic_param['data_docano_format']:
                            json.dump(i, f, ensure_ascii=False)



                os.system(
                    'python3 ./NLLP/experiment/PaddleNLP_UIE/doccano.py \
                    --doccano_file ' + STORE_PATH + '/ml_backend_tmp_decanno.jsonl \
                    --task_type ext \
                    --save_dir ' + STORE_PATH + '\
                    --seed 20230321 \
                    --splits 1 0 0 \
                    --schema_lang ch \
                    --negative_ratio 3 '
                )

                os.system(
                    'export finetuned_model=' + MODEL_PATH
                )

                empty_cache()
                gc.collect()

                
                

                os.system(
                    "python3 ./NLLP/experiment/PaddleNLP_UIE/finetune.py  \
                        --device gpu \
                        --logging_steps 10 \
                        --save_steps 100 \
                        --eval_steps 100 \
                        --seed 87 \
                        --model_name_or_path uie-base  \
                        --output_dir " + MODEL_PATH +"\
                        --train_path " + STORE_PATH + "/train.txt \
                        --dev_path " + STORE_PATH + "/dev.txt  \
                        --max_seq_length 512  \
                        --per_device_eval_batch_size " + str(BATCH_SIZE) + "\
                        --per_device_train_batch_size " + str(BATCH_SIZE) + " \
                        --num_train_epochs 1 \
                        --learning_rate 1e-5 \
                        --label_names 'start_positions' 'end_positions' \
                        --do_train \
                        --do_eval \
                        --do_export \
                        --export_model_dir " + MODEL_PATH + " \
                        --overwrite_output_dir \
                        --disable_tqdm True \
                        --metric_for_best_model eval_f1 \
                        --load_best_model_at_end  True \
                        --save_total_limit 1"
                )
                
                self.dynamic_param['data_docano_format'] = []
            self.write_dynamic_records()
            #text = kwargs['data']['task']['data']['text']
            
        print("============== end of fit() ===============")
        return {'path': workdir}
