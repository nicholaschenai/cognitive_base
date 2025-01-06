from ..voyager_coder.make_procedural import make_voyager_procedural
from ..voyager_coder.base_vector_mem import BaseVectorMem

from ...utils.formatting import tag_indent_format


class CoalaProceduralMem(make_voyager_procedural(BaseVectorMem)):
    """
    Attributes:
        resume (bool): Flag to indicate whether to resume from saved state.
        plan (bool): Whether to include plan in code retrieval.
        rule_db (Database): Database instance for storing and retrieving rules.
        examples_analyzed (set): Set of db examples that have been analyzed for rule extraction.
    """
    def __init__(
        self,
        retrieval_top_k=5,
        ckpt_dir="ckpt",
        vectordb_name="skill",
        resume=False,
        **kwargs,
    ):
        """
        Parameters:
            retrieval_top_k (int): The number of top results to retrieve in search queries.
            ckpt_dir (str): Directory path for checkpoints.
            vectordb_name (str): Name of the vector database for skills.
            resume (bool): Flag to indicate whether to resume from saved state.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            retrieval_top_k=retrieval_top_k,
            ckpt_dir=ckpt_dir,
            vectordb_name=vectordb_name,
            resume=resume,
            **kwargs,
        )

        self.register_vectordb_with_methods('non_func')

    """
    Retrieval Actions (to working mem)
    """
    def _transform_code(self, doc_obj):
        """
        Transforms a document object into code content by extracting from fn_str_map.
        
        Args:
            doc_obj: The document object containing metadata.
            
        Returns:
            str: The code content from fn_str_map.
        """
        code = self.fn_str_map[doc_obj.metadata['name']]['code']
        return doc_obj.page_content + '\n\n' + code
    
    def retrieve_code(self, query, **kwargs):
        """
        Retrieves code snippets based on a query and formats them.

        Args:
            query (str): The query string for retrieval.
            **kwargs: Additional arguments passed to the retrieval method.

        Returns:
            Union[List[str], List[Tuple[str, float]]]: Either:
                - A list of formatted code snippets with proper indentation
                - A list of tuples (formatted_string, score) if scores were requested
        """
        return self._retrieve_and_format(
            query,
            'vector',
            'Callable Code',
            transform_fn=self._transform_code,
            **kwargs
        )
    
    def retrieve_non_func(self, query, **kwargs):
        return self._retrieve_and_format(
            query, 
            'non_func', 
            'Reference Code (Not callable)', 
            transform_fn=transform_non_func, 
            **kwargs
    )

    """
    Learning Actions (from working mem)
    """
    def add_skill(self, parsed_result, skill_description, task, metadata_map=None, task_id=None):
        """
        Adds a skill to the procedural memory, either as a regular function or as a rule,
        based on the presence of a program name and parent information in the parsed result.

        Parameters:
            parsed_result (dict): Parsed result
            skill_description (str): Description of the skill.
            task (str): Task associated with the skill.
            metadata_map (dict, optional): mapping of metadata keys before adding to db.
        """
        if parsed_result['program_name'] and parsed_result['no_parent']:
            # regular function like voyager
            super().add_skill(parsed_result, skill_description, task, metadata_map=metadata_map)
        else:
            # no standalone fn
            metadata = {
                'task': task_id if task_id is not None else task,
                'code': parsed_result['program_code'],
            }
            self.update_methods['non_func'].update(skill_description, metadata=metadata)


def transform_non_func(doc_obj):
    """
    Transforms a document object into code content by extracting from fn_str_map.
    
    Args:
        doc_obj: The document object containing metadata.
        
    Returns:
        str: The code content from fn_str_map.
    """
    code = doc_obj.metadata['code']
    return doc_obj.page_content + '\n\n' + code