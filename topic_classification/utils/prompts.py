import ujson

prompts_dict = {
    "data_source": {
        "zh": "你是一个数据来源标注专家，现有以下五个数据来源：互联网，书籍，学术论文，代码，问答题。其中问答题可能涉及编程或学术内容。请判断以下文本片段属于哪个来源，直接给出来源名，不要有其他表述:",
        "en": "You are a data source annotation expert, and you currently have the following five data sources: the internet, books, academic papers, code, and question and answer tasks. The question and answer tasks may involve programming or academic content. Please determine which source the following text fragment belongs to and simply provide the name of the source, without any other expressions:" 
    },
    "topic": { # topic classification
        "zh": "你是一个数据话题标注专家，现有以下六个数据话题：法律，医药，经济，理工科学，代码，其他。请判断以下文本片段属于哪个话题，直接给出话题名，不要有其他表述:",
        "en": "You are a data topic annotation expert, and you currently have the following six data topics: law, medical && health care, finance, science, code, and other. Please determine which topic the following text fragment belongs to and simply provide the name of the topic, without any other expressions:"
    },
    "topic_props": { # topic probablities (prompt 待修改)
        "zh": "你是一个数据话题标注专家，现有以下六个数据话题：法律，医药，经济，理工科学，代码，其他。请判断以下文本片段属于哪个话题，直接给出话题名，不要有其他表述:",
        "en": "You are a data topic annotation expert, and you currently have the following six data topics: law, medical && health care, finance, science, code, and other. Please determine which topic the following text fragment belongs to and simply provide the name of the topic, without any other expressions:"
    }
}

