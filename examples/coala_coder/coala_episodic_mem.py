from typing import Dict

from ...memories.episodic.base_episodic_mem import BaseEpisodicMem

from ...utils.formatting import tag_indent_format


class CoalaEpisodicMem(BaseEpisodicMem):
    def __init__(
        self,
        retrieval_top_k=5,
        ckpt_dir="ckpt",
        **kwargs,
    ):
        super().__init__(
            retrieval_top_k=retrieval_top_k,
            ckpt_dir=ckpt_dir,
            **kwargs,
        )

    def _format_transition_str(self, transition: Dict):
        """
        Format a transition into a readable string with the following components:
        - Task description
        - Previous critique (if exists)
        - Raw message (thought process and code attempt)
        - Environment observation/feedback
        - Success/failure status
        """
        # Format each component in order with explicit None values
        components = []
        
        # Always show task
        task = transition.get('task', 'None')
        components.append(('Task', task))
        
        # Show all other components, with explicit None for missing values
        components = [
            ('Task', transition.get('task', 'None')),
            ('Previous Critique', transition.get('critique', 'None')),
            ('Thought Process and Code', transition.get('raw_msg', 'None')),
            ('Environment Feedback', transition.get('obs', 'None')),
        ]
        
        # Always show status
        status = 'Success' if transition.get('reward', False) else 'Failure'
        components.append(('Result', status))
        
        # Format each component with its tag
        formatted_str = ''
        for tag, content in components:
            formatted_str += tag_indent_format(tag, [content])
        
        # Wrap the entire transition in a memory tag
        # return tag_indent_format('Past Memory', [formatted_str])
        return formatted_str
