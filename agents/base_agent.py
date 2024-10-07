import logging

# TODO: have a feeling this is circular import
from agent_expt_suite.envs import env_globals

from ..utils.log import train_ckpt

logger = logging.getLogger("logger")


class BaseAgent:
    """
    BaseAgent serves as the foundational class for agents within the cognitive architecture,
    facilitating the integration of various memory modules and decision-making capabilities.
    It is designed to manage the lifecycle of training and evaluating agents on coding tasks.

    Attributes:
        args (dict): Configuration arguments for the agent.
        max_train_iter (int): Maximum number of training iterations.
        handler (object): Handler for managing task-specific operations.
        train_iter (int): Current training iteration.
        max_task_attempts (int): Maximum attempts allowed per task.
        task_id (str): Identifier for the current task.
        train (bool): Flag indicating if the agent is in training mode.
        attr_to_save (list): Attributes to be saved for persistence.
        eval_later (bool): Flag indicating if evaluation should be deferred.
        use_public_tests (bool): Flag indicating if public tests are to be used.
        debug_mode (bool): Flag indicating if the agent is in debug mode.
        verbose (bool): Flag indicating if verbose logging is enabled.
        task (str): Description of the current task.
        procedural_mem (object): Procedural memory module.
        semantic_mem (object): Semantic memory module.
        episodic_mem (object): Episodic memory module.

    Methods:
        print_doc_count(): Prints the document count from each memory module if verbose logging is enabled.
        reset(full_task): Resets the agent's state for a new task.
        train_step(): Abstract method for implementing a single training step.
        train_loop(): Abstract method for implementing the training loop.
    """
    def __init__(self, args, components, handler):
        # params
        self.args = vars(args)

        # logging
        self.handler = handler

        # params
        self.max_task_attempts = args.max_attempts_per_task
        self.max_train_iter = args.max_train_iter

        # checkpointing
        self.task_id = None
        self.train_iter = 0
        self.attr_to_save = ['train_iter', 'task', 'task_id']

        # eval
        self.train = True
        self.eval_later = args.eval_later
        self.use_public_tests = args.use_public_tests

        # debug n print
        self.debug_mode = args.debug_mode
        self.verbose = args.verbose

        # holds the full task details esp those relevant for eval in env
        self.full_task = {}

        """
        Working Memory: short term memory reflecting current circumstances
        """
        self.task = ''

        """
        Long term memory modules
        """
        self.procedural_mem = components.get('procedural_mem', None)
        self.semantic_mem = components.get('semantic_mem', None)
        self.episodic_mem = components.get('episodic_mem', None)

        """
        Reasoning modules
        """

    """
    helper fns
    """
    def print_doc_count(self):
        """
        Prints the document count from each memory module if verbose logging is enabled.
        This method is useful for debugging and understanding the memory utilization.
        """
        for mem in [self.procedural_mem, self.semantic_mem, self.episodic_mem]:
            if mem:
                mem.print_doc_count()

    def reset(self, full_task):
        """
        Resets the agent's state for a new task, including task ID and description,
        and invokes the environment's reset method with the full task details.

        Args:
            full_task (dict): Dictionary containing details of the task to reset to.
        """
        self.full_task = full_task
        task_id = full_task.get('task_id', str(self.train_iter))
        self.handler.task_id = task_id
        self.task_id = task_id
        self.task = full_task['task']
        env_globals.task_env.reset(full_task)

    """
    Decision procedures: Pick which actions to take
    Usually retrieval + reasoning to choose grounding / learning
    """
    def train_step(self):
        """
        Abstract method for implementing a single training step.
        This method should be overridden by subclasses to define the training logic.
        """
        pass

    def train_loop(self):
        while True:
            # setup
            if self.train_iter >= self.max_train_iter:
                print("max train iter reached")
                break
            logger.info(f'[train iter]: {self.train_iter}/{self.max_train_iter} \n')

            # fill ids first for logging before task reset (eg curriculum still deciding task)
            self.task_id = self.train_iter
            self.handler.task_id = self.train_iter

            # fill in your train step
            self.train_step()

            # postprocessing
            self.train_iter += 1
            train_ckpt(self)
            if self.verbose:
                self.print_doc_count()
