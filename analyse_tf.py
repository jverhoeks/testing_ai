from llama_index.core import SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader
from llama_index.core import Document
from pydantic import BaseModel, Field
from typing import List
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.llms.bedrock import Bedrock
import logging 
import json 

logging.basicConfig(level=logging.INFO)


# response model
class TerraformInsights(BaseModel):
    """Insights extraction from Terraform"""

    summary: str
    resources: List[str]
    count_modules: int 
    count_variables: int 
    count_resources: int 


# Terraform file custom reader
class TFReader(BaseReader):
    def load_data(self, file, extra_info=None):
        with open(file, "r") as f:
            text = f.read()
        # load_data returns a list of Document objects
        return [Document(text=text, extra_info=extra_info or {})]


logging.info("Loading LLM model")
llm = Bedrock(
    model="anthropic.claude-3-haiku-20240307-v1:0",
    temperature=0,
    profile_name = "bedrockai"
)


tf_source = '../tfcode/'

logging.info("Reading Terraform files %s", tf_source)
reader = SimpleDirectoryReader(
    input_dir=tf_source, file_extractor={".tf": TFReader()},
    required_exts=[".tf"],
    recursive=True
)
tf_documents = reader.load_data()

logging.info('Found %i files', len(tf_documents))






prompt_template_str = """\
Generate a summary and count the resources in the following terraform file: \
{document}\
"""


tf_documents = tf_documents[1:4]
logging.info('Running with subset of  %i files', len(tf_documents))

logging.info('Calling LLMTextCompletionProgram')
program = LLMTextCompletionProgram.from_defaults(
    llm=llm,
    output_cls=TerraformInsights,
    prompt_template_str=prompt_template_str,
    verbose=True,
)

for tf_document in tf_documents:
    output = program(document=tf_document)
    print(json.dumps(dict(output),indent=2))



