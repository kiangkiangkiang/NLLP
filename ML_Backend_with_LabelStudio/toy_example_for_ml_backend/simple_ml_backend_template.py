#將此block複製到./toy_example_for_ml_backend/simple_ml_backend_template.py內即可
from label_studio_ml.model import LabelStudioMLBase
from paddlenlp import Taskflow
import numpy as np

def postprocessing(predict_result, all_relations):
    predict_result = sorted(predict_result, key=lambda x: x['value']['start'])

    #Delete Overlap Entity
    index = 0
    while index < len(predict_result) - 1:
        if predict_result[index]['value']['start'] == predict_result[index + 1]['value']['start'] :
            if predict_result[index]['value']['score'] > predict_result[index + 1]['value']['score']:
                predict_result.pop(index + 1)
            else:
                predict_result.pop(index)
        else:
            index += 1

    #Delete Repeat Person
    unique_name_map, result_index = [], []
    for i, item in enumerate(predict_result):
        if item['value']['labels'][0] == '原告':
            if item['value']['text'] not in unique_name_map:
                unique_name_map.append(item['value']['text'])
                result_index.append(i)
        else:
            result_index.append(i)
    predict_result = list(np.array(predict_result)[result_index])

    #Add Relation
    start_id_mapping = {predict_result[index]['value']['start']: predict_result[index]['id'] for index in range(len(predict_result))}
    for relation in all_relations:
        people_start = relation['start']
        for relation_type in relation['relations']:
            money_start = relation['relations'][relation_type][0]['start']
            if start_id_mapping.get(people_start) and start_id_mapping.get(money_start):
                predict_result.append({
                    'from_id': start_id_mapping[people_start],
                    'to_id': start_id_mapping[money_start],
                    "type": "relation",
                    "direction": "right",
                    "labels": [
                        relation_type
                    ]
                })
    return predict_result

class SimpleMLBackend(LabelStudioMLBase):
    #載入模型
    def __init__(self, **kwargs) -> None:
        super(SimpleMLBackend, self).__init__(**kwargs)
        my_toy_schema = ['非財產上之損害的求償精神慰撫金', '法院得心證判斷之適當裁定精神慰撫金',
                        {'原告': '非財產上之損害的求償精神慰撫金',
                         '原告': '法院得心證判斷之適當裁定精神慰撫金'}]
        self.model = Taskflow("information_extraction", schema=my_toy_schema)

        #RE任務需用ID連接，因此每個Entitu都定義ID
        self.entity_id = 0
    

    #Model Inference
    def predict(self, tasks, **kwargs):
        print("Start Label...")
        predictions = []
        for task in tasks:
            #task 格式同「Show task source」
            uie_output = self.model(task['data']['text'])[0]
            result, relations = [], []
            for key in uie_output:
                for item in uie_output[key]:
                    #result對齊Label Studio標好的格式                    
                    result.append({
                        'value': {
                            'start': item['start'],
                            'end': item['end'],
                            'score': item['probability'],
                            'text': item['text'],
                            'labels': [key]
                        },
                        'id': str(self.entity_id),
                        'from_name': 'label', 
                        'to_name': 'text',
                        'type': 'labels'
                    })
                    self.entity_id += 1

                    if item.get('relations') is not None:
                        relations.append(item)

            result = postprocessing(predict_result=result, all_relations=relations)
            predictions.append({
                'result': result,
                'model_version': 'my_toy_example_using_uie'
            })

        print("End Label...")
        return predictions


    #NOT IMPLEMENT FOR THIS CASE
    def fit(self, annotation, workdir=None, **kwargs):
        return