"""
Example CoALA coding agent as described in the CoALA paper.

> For example, a future coding agent could maintain human-provided programming knowledge (semantic) such as 
> manuals, textbooks, problems, and examples, as well as its problem solutions and test records (episodic), 
> reflections, and summaries on top of these experiences (semantic), and a gradually expanding code library 
> that stores useful methods, e.g., QuickSort, GCD, LCA (procedural).
"""
import logging

from typing import Dict, Any, Tuple
from argparse import Namespace

from ...agents.base_agent import BaseAgent

from .coala_episodic_mem import CoalaEpisodicMem
from .coala_procedural_mem import CoalaProceduralMem
from ...memories.semantic.base_semantic_mem import BaseSemanticMem

from .coala_reasoning import CoalaReasoning
from .coala_desc import CoalaDesc

from ...utils.log import handle_rollout_error
from ...utils.code_parse import append_dependencies
from ...utils.formatting import truncate_str

from .utils import TaskLogger
from .utils.coala_message_thread import CoalaMessageThread

logger = logging.getLogger("logger")


class CoalaAgent(BaseAgent):
    def __init__(self, args: Namespace, components: Dict[str, Any], handler: Any):
        super().__init__(args, components, handler)

        # misc
        self.data_pipeline = None

        # settings
        self.agent_type = args.agent_type

        # params
        self.retrieval_top_k = args.retrieval_top_k

        """
        Working Memory: short term memory reflecting current circumstances
        """

        """
        Long term memory modules
        """
        self.procedural_mem = CoalaProceduralMem(**self.args)
        self.episodic_mem = CoalaEpisodicMem(**self.args)
        self.semantic_mem = BaseSemanticMem(**self.args)

        """
        Reasoning modules
        """
        self.reasoning_module = CoalaReasoning(callbacks=[handler], **self.args)
        self.desc_module = CoalaDesc(callbacks=[handler], **self.args)

        """
        decision procedure modules
        """

    """
    helpers
    """
    def reset(self, full_task):
        """Reset working memory between tasks"""
        super().reset(full_task)
        self.reasoning_module.reset(full_task)

    def get_next_task(self) -> dict:
        if self.data_pipeline is None:
            raise ValueError("Data pipeline not properly attached to agent")
        full_task = self.data_pipeline.get_next_task(self.train_iter)
        return full_task

    def retrieve_for_coding(self, cue):
        if self.agent_type == 'react':
            return []
        
        # Get all retrievals with scores
        all_retrievals = []
        
        # Episodic memory retrieval
        episodic_results = self.episodic_mem.retrieve_transition(cue, with_scores=True)
        all_retrievals.extend(episodic_results)
        
        # Semantic memory retrievals
        textbook_results = self.semantic_mem.retrieve_textbook(cue, with_scores=True)
        reflection_results = self.semantic_mem.retrieve_reflections(cue, with_scores=True)
        summary_results = self.semantic_mem.retrieve_summaries(cue, with_scores=True)
        
        all_retrievals.extend(textbook_results)
        all_retrievals.extend(reflection_results)
        all_retrievals.extend(summary_results)
        
        # Procedural memory retrievals
        code_results = self.procedural_mem.retrieve_code(cue, with_scores=True)
        non_func_results = self.procedural_mem.retrieve_non_func(cue, with_scores=True)
        
        all_retrievals.extend(code_results)
        all_retrievals.extend(non_func_results)
        
        # Sort all retrievals by score (ascending since smaller L2 distance is better)
        sorted_retrievals = sorted(all_retrievals, key=lambda x: x[1])
        
        # Log top k retrievals with truncated strings
        for content, score in sorted_retrievals[:self.retrieval_top_k]:
            logger.info(f"Retrieved (score={score:.4f}): {truncate_str(content)}")
        
        # Return only the content (first element) of top k tuples
        return [item[0] for item in sorted_retrievals[:self.retrieval_top_k]]

    def process_transition(self, transition_info: Dict, attempt_idx: int):
        """Process a transition in memory"""
        if not self.train:
            return

        transition_info['transition_id'] = attempt_idx
        transition_info['task'] = self.task
        transition_info['task_id'] = self.task_id
        self.episodic_mem.add_transition(transition_info)

    def rollout(self, full_task, use_public_tests=False) -> Tuple[bool, Dict, CoalaMessageThread]:
        """Execute a rollout of the task"""
        self.reset(full_task)
        
        obs, reward, info = '', False, {}
        parsed_result = {}
        message_thread = CoalaMessageThread()
        
        logger.info(f'Attempting task_id {self.task_id}')
        task_logger = TaskLogger(self.result_dir, self.train, self.task_id)
        
        try:
            # Initial solve attempt - only for retrieval purposes
            context, message_thread = self.reasoning_module.initial_solve()
            
            for attempt_idx in range(self.max_task_attempts):
                logger.info(f"\033[35m Rollout attempt {attempt_idx+1}/{self.max_task_attempts}\033[0m")

                if attempt_idx > 0:
                    context = self.reasoning_module.critique(message_thread)
                
                retrieved = self.retrieve_for_coding(context)
                
                parsed_result = self.reasoning_module.gen_code(message_thread, retrieved, attempt_idx)

                if not parsed_result:
                    continue

                full_code = append_dependencies(parsed_result, self.procedural_mem.fn_str_map)
                message_thread.update_full_code(full_code)
                
                if self.eval_later and not self.train and not use_public_tests:
                    break

                obs, reward, _, info = self.env_interface.step(full_code, use_public_tests)
                message_thread.record_env_out(obs, reward, info)
                
                transition_info, iteration_data = message_thread.get_latest_data()
                
                self.process_transition(transition_info, attempt_idx)
                task_logger.log_iteration(iteration_data, attempt_idx)

                if reward:
                    break
        except Exception as e:
            handle_rollout_error(e, self.task_id, self.result_dir)

        self.episodic_mem.finish_episode()
        
        task_logger.log_rollout(message_thread.get_rollout_data())        
        return reward, parsed_result, message_thread

    """
    decision procedures
    """
    def train_step(self):
        """Execute a training step"""
        full_task = self.get_next_task()
        success, parsed_result, message_thread = self.rollout(full_task)

        # Generate and record training data
        summary = self.reasoning_module.summarize(message_thread)
        reflection = self.reasoning_module.reflect(message_thread, full_task.get('code', ''))

        # Update semantic memory
        self.semantic_mem.update_summaries(summary, metadata={'task_id': self.task_id})
        self.semantic_mem.update_reflections(reflection, metadata={'task_id': self.task_id})
        
        if success:
            desc = self.desc_module.run(parsed_result, self.task)
            message_thread.record_desc(desc)
            self.procedural_mem.add_skill(parsed_result, desc, self.task, task_id=self.task_id)

        # Save training data
        training_data = message_thread.get_training_data()
        task_logger = TaskLogger(self.result_dir, True, self.task_id)
        task_logger.log_train(training_data)

    def test_one(self, full_task):
        success, parsed_result, _ = self.rollout(full_task, self.use_public_tests)
        return success, parsed_result
