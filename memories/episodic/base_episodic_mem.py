import copy

from typing import Dict

from ..base_mem import BaseMem, conditional_memory_op

from ...utils import load_json, dump_json


class BaseEpisodicMem(BaseMem):
    """
    Stores a base episode (a sequence of past experiences (transitions))

    We use (state, action, reward, next_state, done) as the transition tuple (with a task header)
    
    Can retrieve transitions, timesteps surrounding a given transition and full episodes

    Principles (see README):
    - Retrieved memory distinguished frm current sensing to prevent confusion
    - Temporal index
    - Automatic (debatable)

    TODO:
    - Filter by time window
    - Matching by symbolic attributes
    - Partial matching (eg part of cue or part of transition)
    - Retrieve sequence of transitions
    """
    def __init__(
        self,
        retrieval_top_k=5,
        ckpt_dir="ckpt",
        vectordb_name="episodic",
        resume=True,
        **kwargs,
    ):
        super().__init__(
            retrieval_top_k=retrieval_top_k,
            ckpt_dir=ckpt_dir,
            vectordb_name=vectordb_name,
            resume=resume,
            **kwargs,
        )

        # ablations
        self.is_enabled = not kwargs.get('disable_episodic', False)

        self.curr_episode = []
        self.transition_id = 0
        self.episode_id = 0

        if resume:
            self.load_episode_state()

    """
    helper fns
    """
    def load_episode_state(self):
        """Load episode state from checkpoint"""
        state = load_json(f"{self.ckpt_dir}/episodic/episode_state.json")
        
        self.episode_id = state.get("episode_id", 0)
        self.transition_id = state.get("transition_id", 0)
        self.curr_episode = state.get("curr_episode", [])

        if state:
            print(f"\033[35mLoaded episode state: episode_num={self.episode_id}\033[0m")
        else:
            print(f"\033[35mNo episode state found, starting fresh\033[0m")

    def save_episode_state(self):
        """Save episode state to checkpoint"""
        state = {
            "episode_id": self.episode_id,
            "transition_id": self.transition_id,
            "curr_episode": self.curr_episode,
        }
        dump_json(state, f"{self.ckpt_dir}/episodic/episode_state.json", indent=4)
    
    def finish_episode(self):
        """Mark current episode as complete and increment episode counter"""
        # only add transitions at the end of an episode to prevent retrieval of existing transitions
        for transition_str, metadata in self.curr_episode:
            self.update(transition_str, metadata=metadata)
        print(f"\033[35mFinished episode {self.episode_id}\033[0m")
        self.curr_episode = []
        self.transition_id = 0
        self.episode_id += 1
        self.save_episode_state()
        
    def _format_transition(self, transition_data: Dict):
        """
        Override this method to customize

        Memory-specific formatting - HOW to store
        e.g., adding required fields, ensuring consistency

        Usually to separate out the stuff to be formatted into str (rest is metadata)
        Returns:
            tuple: (formatted_transition, metadata)
            - formatted_transition: dict containing the transition content
            - metadata: dict containing at minimum transition_id and task_id
        """
        formatted = copy.deepcopy(transition_data)
        formatted.pop('transition_id')
        formatted.pop('task_id')
        return formatted

    def _format_transition_str(self, transition):
        """Override this method to customize string representation"""
        raise NotImplementedError

    """
    Retrieval Actions (to working mem / decision procedure)
    """
    @conditional_memory_op
    def retrieve_transition(self, query, **kwargs):
        """Retrieve transitions based on similarity to query"""
        # return self.retrieve(query, **kwargs)
        return self._retrieve_and_format(query, 'vector', 'Past Memory', **kwargs)

    # TODO: incomplete
    # def retrieve_surrounding_timesteps(self, transition_id, window=1):
    #     """Retrieve transitions before and after a given transition"""
    #     # Get episode containing the transition
    #     result = self.vectordb.get(
    #         where={"transition_id": transition_id}
    #     )
    #     if not result['metadatas']:
    #         return []
        
    #     episode_id = result['metadatas'][0]['episode_id']
    #     transition_idx = result['metadatas'][0]['transition_idx']
        
    #     # Get surrounding transitions
    #     start_idx = max(0, transition_idx - window)
    #     end_idx = transition_idx + window + 1
        
    #     surrounding = self.vectordb.get(
    #         where={
    #             "$and": [
    #                 {"episode_id": episode_id},
    #                 {"transition_idx": {"$gte": start_idx}},
    #                 {"transition_idx": {"$lt": end_idx}}
    #             ]
    #         }
    #     )
    #     return surrounding['documents']

    # TODO: implement episode retrieval by id
    # def get_episode(self, episode_id=None):
    #     """
    #     Retrieve a full episode by ID or current episode
        
    #     Currently only implements latter
    #     """
    #     if episode_id is None:
    #         return format_episode_str(self.curr_episode)
        
    #     result = self.vectordb.get(
    #         where={"episode_id": episode_id}
    #     )
    #     if not result['metadatas']:
    #         return ""
        
    #     transitions = [
    #         {**meta, 'content': doc}
    #         for doc, meta in zip(result['documents'], result['metadatas'])
    #     ]
    #     transitions.sort(key=lambda x: x['transition_idx'])
    #     return format_episode_str(transitions)

    """
    Learning Actions (from working mem)
    """
    @conditional_memory_op
    def add_transition(self, transition_data):
        """Add a transition to the current episode
        
        Args:
            transition_data: Dictionary containing all transition information
                (already formatted according to agent's strategy, including any task info)
                combine multiple sources in the agent's strategy
        """
        self.transition_id = transition_data['transition_id']
        # Memory-specific formatting (e.g., ensuring correct structure, adding memory-specific metadata)
        transition = self._format_transition(transition_data)
        # self.curr_episode.append(transition)

        # Format transition for storage
        transition_str = self._format_transition_str(transition)
        
        # Combine base metadata with episode metadata
        transition_data_copy = copy.deepcopy(transition_data)
        transition_data_copy['episode_id'] = self.episode_id

        # self.update(transition_str, metadata=transition_data_copy)
        self.curr_episode.append((transition_str, transition_data_copy))
        self.save_episode_state()
        
        # self.transition_id += 1
        # if self._is_episode_done(transition):
        #     self.finish_episode()


# TODO: incomplete
# def format_episode_str(transitions):
#     """Format a list of transitions into a readable episode string"""
#     episode = ''
#     if transitions and transitions[0].get('task_header'):
#         episode += f"Task: {transitions[0]['task_header']}\n"
#     for i, t in enumerate(transitions):
#         episode += f"\nTransition {i}:\n"
#         episode += format_transition_str(t)
#     return episode
