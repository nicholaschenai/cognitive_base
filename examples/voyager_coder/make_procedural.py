"""
stores procedures which implement actions (grounding, reasoning, learning, retrieval) or decision making
eg skill library

This implementation is heavily adapted from Voyager: An Open-Ended Embodied Agent with Large Language Models by
Wang et. al. (2023) https://github.com/MineDojo/Voyager
"""
from typing import Type, TypeVar

from .base_vector_mem import BaseVectorMem

from ...utils import dump_text, f_mkdir

T = TypeVar('T', bound=BaseVectorMem)


# Make this into a fn to reuse elsewhere
# class VoyagerProceduralMem(BaseVectorMem):
def make_voyager_procedural(base_class: Type[T]) -> Type[T]:
    class VoyagerProcedural(base_class):
        def __init__(
            self,
            retrieval_top_k=5,
            ckpt_dir="ckpt",
            vectordb_name="skill",
            resume=False,
            no_skill_files=False,
            **kwargs,
        ):
            super().__init__(
                retrieval_top_k=retrieval_top_k,
                ckpt_dir=ckpt_dir,
                vectordb_name=vectordb_name,
                resume=resume,
                **kwargs,
            )
            self.no_skill_files = no_skill_files
            if not no_skill_files:
                f_mkdir(f"{ckpt_dir}/{vectordb_name}/description")
                f_mkdir(f"{ckpt_dir}/skill/code")

            self._check_vectordb_sync(
                f"Skill Manager's vectordb is not synced with entries.json.\n"
                f"There are {self.vectordb._collection.count()} skills in vectordb but \n"
                f"{len(self.fn_str_map)} skills in entries.json.\n"
                f"Did you set resume=False when initializing the manager?\n"
                f"You may need to manually delete the vectordb directory for running from scratch."
            )

        """
        helper fns
        """

        """
        Retrieval Actions (to working mem / decision procedure)
        """

        """
        Learning Actions (from working mem)
        """
        def add_skill(self, parsed_result, skill_description, task, metadata_map=None):
            """
            Adds a new skill (code) to the database.

            Args:
                parsed_result (dict): The parsed result containing skill information.
                skill_description (str): A description of the skill.
                task (str): The task associated with the skill.
                metadata_map (list, optional): A list of tuples mapping source keys to destination keys. Defaults to None.

            Returns:
                str: The name of the added skill.
            """
            program_code = parsed_result.get("program_code", "")

            result_assertion = program_code and ("program_name" in parsed_result)
            assert result_assertion, "program, program_name must be returned"

            parsed_result["task"] = task

            mapping = [
                ("program_code", "code"),
                ("program_name", "name"),
                ("dependencies", "dependencies"),
                ("task", "task")
            ]
            if metadata_map:
                mapping += metadata_map

            dumped_prog_name = self.add_code(parsed_result, mapping, skill_description)

            if self.no_skill_files:
                return
            dump_text(program_code, f"{self.ckpt_dir}/skill/code/{dumped_prog_name}.py")
            dump_text(skill_description, f"{self.ckpt_dir}/skill/description/{dumped_prog_name}.txt")

    return VoyagerProcedural
