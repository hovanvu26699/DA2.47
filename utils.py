
def read_question(file):
    with open(file,'r',encoding='utf-8') as f:
        question = f.readlines()
    question = [x.strip() for x in question]
    return question

def read_docs(file):
    with open(file,'r',encoding='utf-8') as f:
        docs = f.read()
    return docs

if __name__ =="__main__":
    print(read_question("./data/question.txt"))
    print(read_docs("./data/docs.txt"))