# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import requests
from fastapi.responses import StreamingResponse
from langchain.chains.summarize import load_summarize_chain
# from langchain.chains.llm_requests import load_llm_requests_chain
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer

from comps import CustomLogger, GeneratedDoc, OpeaComponent, ServiceType
from comps.cores.mega.utils import ConfigError, get_access_token, load_model_configs
from comps.cores.proto.api_protocol import DocSumChatCompletionRequest

from langchain.chains import LLMChain

from .template import templ_en, templ_refine_en, templ_refine_zh, templ_zh

logger = CustomLogger("llm_docsum")
logflag = os.getenv("LOGFLAG", False)

# Environment variables
MODEL_NAME = os.getenv("LLM_MODEL_ID")
MODEL_CONFIGS = os.getenv("MODEL_CONFIGS")
TOKEN_URL = os.getenv("TOKEN_URL")
CLIENTID = os.getenv("CLIENTID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", 2048))
MAX_TOTAL_TOKENS = int(os.getenv("MAX_TOTAL_TOKENS", 4096))

if os.getenv("LLM_ENDPOINT") is not None:
    DEFAULT_ENDPOINT = os.getenv("LLM_ENDPOINT")
elif os.getenv("TGI_LLM_ENDPOINT") is not None:
    DEFAULT_ENDPOINT = os.getenv("TGI_LLM_ENDPOINT")
elif os.getenv("vLLM_ENDPOINT") is not None:
    DEFAULT_ENDPOINT = os.getenv("vLLM_ENDPOINT")
else:
    DEFAULT_ENDPOINT = "http://localhost:8080"


def get_llm_endpoint():
    if not MODEL_CONFIGS:
        return DEFAULT_ENDPOINT
    else:
        # Validate and Load the models config if MODEL_CONFIGS is not null
        configs_map = {}
        try:
            configs_map = load_model_configs(MODEL_CONFIGS)
        except ConfigError as e:
            logger.error(f"Failed to load model configurations: {e}")
            raise ConfigError(f"Failed to load model configurations: {e}")
        try:
            return configs_map.get(MODEL_NAME).get("endpoint")
        except ConfigError as e:
            logger.error(f"Input model {MODEL_NAME} not present in model_configs. Error {e}")
            raise ConfigError(f"Input model {MODEL_NAME} not present in model_configs")


class OpeaDocSum(OpeaComponent):
    """A specialized OPEA DocSum component derived from OpeaComponent.

    Attributes:
        client (TGI/vLLM): An instance of the TGI/vLLM client for text generation.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.LLM.name.lower(), description, config)
        self.access_token = (
            get_access_token(TOKEN_URL, CLIENTID, CLIENT_SECRET) if TOKEN_URL and CLIENTID and CLIENT_SECRET else None
        )
        self.llm_endpoint = get_llm_endpoint()
        
        print("MODEL_NAME ====>> ", MODEL_NAME)
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaDocSum health check failed.")

    async def generate(self, input: DocSumChatCompletionRequest, client):
        """Invokes the TGI/vLLM LLM service to generate summarization for the provided input.

        Args:
            input (DocSumChatCompletionRequest): The input text(s).
            client: TGI/vLLM based client
        """
        ### get input text
        message = None
        if isinstance(input.messages, str):
            message = input.messages
        else:  # List[Dict]
            for input_data in input.messages:
                if "role" in input_data and input_data["role"] == "user" and "content" in input_data:
                    message = input_data["content"]
                    if logflag:
                        logger.info(f"Get input text:\n {message}")
        if message is None:
            logger.error("Don't receive any input text, exit!")
            return GeneratedDoc(text=None, prompt=None)
        
        print("========================================")
        print("input.custom_prompt", input.custom_prompt)
        print("========================================")
        
        if input.custom_prompt is None:            
            ### check summary type
            summary_types = ["auto", "stuff", "truncate", "map_reduce", "refine"]
            if input.summary_type not in summary_types:
                raise NotImplementedError(f"Please specify the summary_type in {summary_types}")
            if input.summary_type == "auto":  ### Check input token length in auto mode
                token_len = len(self.tokenizer.encode(message))
                if token_len > MAX_INPUT_TOKENS + 50:
                    input.summary_type = "refine"
                    if logflag:
                        logger.info(
                            f"Input token length {token_len} exceed MAX_INPUT_TOKENS + 50 {MAX_INPUT_TOKENS+50}, auto switch to 'refine' mode."
                        )
                else:
                    input.summary_type = "stuff"
                    if logflag:
                        logger.info(
                            f"Input token length {token_len} not exceed MAX_INPUT_TOKENS + 50 {MAX_INPUT_TOKENS+50}, auto switch to 'stuff' mode."
                        )

            ### Check input language
            if input.language in ["en", "auto"]:
                templ = templ_en
                templ_refine = templ_refine_en
            elif input.language in ["zh"]:
                templ = templ_zh
                templ_refine = templ_refine_zh
            else:
                raise NotImplementedError('Please specify the input language in "en", "zh", "auto"')

            ## Prompt
            PROMPT = PromptTemplate.from_template(templ)
            
            print("========================================")
            print("templ::::", templ)
            print("templ_refine::::", templ_refine)
            print("========================================")
            print("PROMPT:::", PROMPT)
            print("========================================")
            print("summary_type:::", input.summary_type)
            print("========================================")
            
            
            if input.summary_type == "refine":
                PROMPT_REFINE = PromptTemplate.from_template(templ_refine)
                print("========================================")
                print("PROMPT_REFINE:::", PROMPT_REFINE)
                print("========================================")
            
            
            if logflag:
                logger.info("After prompting:")
                logger.info(PROMPT)
                if input.summary_type == "refine":
                    logger.info(PROMPT_REFINE)

            ## Split text
            if input.summary_type == "stuff":
                text_splitter = CharacterTextSplitter()
            else:
                if input.summary_type == "refine":
                    if (
                        MAX_TOTAL_TOKENS <= 2 * input.max_tokens + 256 or MAX_INPUT_TOKENS <= input.max_tokens + 256
                    ):  ## 256 is reserved prompt length
                        raise RuntimeError(
                            "In Refine mode, Please set MAX_TOTAL_TOKENS larger than (max_tokens * 2 + 256), MAX_INPUT_TOKENS larger than (max_tokens + 256)"
                        )
                    max_input_tokens = min(
                        MAX_TOTAL_TOKENS - 2 * input.max_tokens - 256, MAX_INPUT_TOKENS - input.max_tokens - 256
                    )
                else:
                    if (
                        MAX_TOTAL_TOKENS <= input.max_tokens + 256 or MAX_INPUT_TOKENS < 256
                    ):  # 256 is reserved token length for prompt
                        raise RuntimeError(
                            "Please set MAX_TOTAL_TOKENS larger than max_tokens + 256, MAX_INPUT_TOKENS larger than 256)"
                        )
                    max_input_tokens = min(MAX_TOTAL_TOKENS - input.max_tokens - 256, MAX_INPUT_TOKENS - 256)
                chunk_size = min(input.chunk_size, max_input_tokens) if input.chunk_size > 0 else max_input_tokens
                chunk_overlap = input.chunk_overlap if input.chunk_overlap > 0 else int(0.1 * chunk_size)
                text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                    tokenizer=self.tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
                if logflag:
                    logger.info(f"set chunk size to: {chunk_size}")
                    logger.info(f"set chunk overlap to: {chunk_overlap}")

            texts = text_splitter.split_text(message)
            docs = [Document(page_content=t) for t in texts]
            if logflag:
                logger.info(f"Split input query into {len(docs)} chunks")
                logger.info(f"The character length of the first chunk is {len(texts[0])}")

            ## LLM chain
            summary_type = input.summary_type
            if summary_type == "stuff":
                llm_chain = load_summarize_chain(llm=client, prompt=PROMPT)
                
                print("========================================")
                print("client::::", client)
                print("========================================")
                print("PROMPT:::", PROMPT)
                print("========================================")
                print("llm_chain:::", llm_chain)
                print("========================================")
                
            
            elif summary_type == "truncate":
                docs = [docs[0]]
                llm_chain = load_summarize_chain(llm=client, prompt=PROMPT)
            elif summary_type == "map_reduce":
                llm_chain = load_summarize_chain(
                    llm=client,
                    map_prompt=PROMPT,
                    combine_prompt=PROMPT,
                    chain_type="map_reduce",
                    return_intermediate_steps=True,
                )
            elif summary_type == "refine":                
                llm_chain = load_summarize_chain(
                    llm=client,
                    question_prompt=PROMPT,
                    refine_prompt=PROMPT_REFINE,
                    chain_type="refine",
                    return_intermediate_steps=True,
                )
            else:
                raise NotImplementedError(f"Please specify the summary_type in {summary_types}")

            if input.stream:

                async def stream_generator():
                    from langserve.serialization import WellKnownLCSerializer

                    _serializer = WellKnownLCSerializer()
                    async for chunk in llm_chain.astream_log(docs):
                        data = _serializer.dumps({"ops": chunk.ops}).decode("utf-8")
                        if logflag:
                            logger.info(data)
                        yield f"data: {data}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(stream_generator(), media_type="text/event-stream")
            else:
                response = await llm_chain.ainvoke(docs)

                if input.summary_type in ["map_reduce", "refine"]:
                    intermediate_steps = response["intermediate_steps"]
                    if logflag:
                        logger.info("intermediate_steps:")
                        logger.info(intermediate_steps)

                output_text = response["output_text"]
                if logflag:
                    logger.info("\n\noutput_text:")
                    logger.info(output_text)

                # return GeneratedDoc(text=output_text, prompt=message)
                results = GeneratedDoc(text=output_text, prompt=message)
            
                print(" =================>>>>> results ::::::", results )
                return results

        else:
                       
            # Create a PromptTemplate to guide the LLM's rewriting
            PROMPT = PromptTemplate.from_template(input.custom_prompt)

            print("========================================")
            print("PROMPT:::", PROMPT )
            print("========================================")
            
            llm_chain = load_summarize_chain(llm=client, prompt=PROMPT)
            # llm_chain = LLMChain(prompt=PROMPT, llm=client)
            
            text_splitter = CharacterTextSplitter()
            texts = text_splitter.split_text(message)
            
            docs = [Document(page_content=t) for t in texts]
            
            response = await llm_chain.ainvoke(docs)
            
            print("========================================")
            print("response::::", response)
            print("========================================")
            
            output_text = response["output_text"]
            
            # output_text = "this is a test place holder for custom prompt"
            
            results = GeneratedDoc(text=output_text, prompt=message)
            
            print("========================================")
            print("results::::", results)
            print("========================================")
                       
            
            return results
        
        
        
            # print("========================================")
            # print("docs::::", docs)
            # print("========================================")
        
            # print("========================================")
            # print("input.stream", input.stream)

            # print("summary_type:::", input.summary_type)
            # print("========================================")
            # print("client::::", client)
            # print("========================================")
            
        
        
        
        
                    # # Define the text you want to rewrite
            # text_to_rewrite = """Intel Corporation, Document ID: TS-INTEL-2023-001, dated October 15, 2023, outlines Project Quantum Leap, 
            #         Intel's initiative to develop advanced quantum computing technology. 
            #         The project aims to build a 1000-qubit processor by late 2025 and improve qubit stability to last over 100 seconds. 
            #         It includes creating a quantum programming language and partnering with universities for algorithm research. 
            #         Intel plans to capture 40% of the quantum computing market by 2030 and collaborate with industry leaders to set global standards. 
            #         The project is divided into three phases: research (early 2023 - late 2024), testing (early 2025 - mid 2026), and launch (mid 2026 - late 2027),
            #         with goals to finish research on quantum error correction, prototype processor design, test processors in controlled settings, 
            #         and start offering quantum computing services globally. The implications include positioning Intel as a leader in new technology, 
            #         potentially changing computing methods, driving economic growth with new applications in security, optimization, and AI, and addressing security 
            #         concerns as quantum computing challenges current encryption standards. Access to this is restricted to personnel with appropriate document clearance, 
            #         and unauthorized disclosure is prohibited.
            #     """

            # # Create a PromptTemplate to guide the LLM's rewriting
            # template = """Rewrite the following text by removing confidential information:

            # "{text}"

            # Rewritten text:"""

            # prompt = PromptTemplate(
            #     input_variables=["text"],
            #     template=template
            # )
            
            
            # # # Create an LLMChain that combines the prompt and the LLM
            # # rewrite_chain = LLMChain(llm=client, prompt=prompt)
            # rewrite_chain = LLMChain(prompt=prompt, llm=client)

            # # # Rewrite the text using the chain
            # # rewritten_text = rewrite_chain.invoke(text_to_rewrite)
            # rewritten_text = await rewrite_chain.ainvoke(text_to_rewrite)

            # # Print the rewritten text
            # print("========================================")
            # print("rewritten_text:::", rewritten_text) 
            # print("========================================")
            
            # results = rewritten_text # GeneratedDoc(text=rewritten_text, prompt=message)
            
            # return results