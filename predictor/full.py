from pyvi import ViTokenizer
class QASystem:
    def __init__(self, retriever,reader):
        self.retriever = retriever
        self.reader = reader
    
    def predict(self, question, retriever_size=1):
        question = ViTokenizer.tokenize(question).replace('_',' ')
        candidate_passages = self.retriever.query(question,retriever_size)
        final_answer = ''
        max_score = 0
        for p in candidate_passages:
            answer, spos, epos, score = self.reader.predict(question,p['text'])[0]
            if answer and score > max_score and spos != 0:
                max_score = score
                final_answer = answer
        return final_answer