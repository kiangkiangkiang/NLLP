def initial_parsed_label_config():
    result = {'label': {
              'type': 'Labels',
              'to_name': ['text'],
              'inputs': [{
                  'type': 'Text',
                  'value': 'text'
              }],
              'labels': ['地名', '人名', "時間"],
              'labels_attrs': {
                  '地名': {
                      'value': '地名',
                      'background': '#FFA39E'
                  },
                  '人名': {
                      'value': '人名',
                      'background': '#D4380D'
                  },
                  '時間': {
                      'value': '時間',
                      'background': '#AD8B00'
                  }
              }
              }
              }
    return result
class MyUIEPredictModel(LabelStudioMLBase):
    def __init__(self, **kwargs) -> None:
        super(MyUIEPredictModel, self).__init__(**kwargs)
        self.parsed_label_config = initial_parsed_label_config()
        self.from_name, self.info = list(self.parsed_label_config.items())[0]
        #print(self.from_name)
        #print(self.info)
        self.to_name = self.info['to_name'][0]
        self.value = self.info['inputs'][0]['value']
        self.labels = list(self.info['labels'])
        print(self.to_name)
        print(self.value)
        print(self.labels)

        #task_paht='./checkpoint/model_best'
        self.model = Taskflow("information_extraction", schema=self.labels)
        
    def predict(self, tasks, **kwargs):
        from_name = self.from_name
        to_name = self.to_name
        model = self.model
        predictions = []
        for task in tasks:
            print("Predict task:", task)
            text = task['data'][self.value]
            uie = model(text)[0]

            result = []
            scores = []
            for key in uie:
                for item in uie[key]:
                    result.append({
                        'from_name': from_name,
                        'to_name': to_name,
                        'type': 'labels',
                        'value':{
                            'start': item['start'],
                            'end': item['end'],
                            'score': item['probability'],
                            'text': item['text'],
                            'labels': [key]
                        }
                    })
                    scores.append(item['probability'])
            result = sorted(result, key=lambda x: x['value']['start'])
            mean_score = np.mean(scores) if len(scores) else 0
            predictions.append({
                'result': result,
                'score': float(mean_score),
                'model_version': 'uie_new'
            })
        return predictions

    def fit(self, annotations, workdir=None, **kwargs):
        print("annotations: ", annotations)
        return 
        
