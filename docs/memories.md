# Memory Elements

## Overview
In cognitive architecture, memory is categorized into three main types based on cognitive science:
- Procedural (skills)
- Semantic (knowledge, facts)
- Episodic (historical states, events etc)

We decouple the elements as much as possible for easier understanding, maintainability and flexibility. They overall look something like this, and each part will be explained later

```python
from retrieval_methods.reflection_retrieval import ReflectionRetrieval
from retrieval_methods.summary_retrieval import SummaryRetrieval

from utils.database.vector_db.chroma_vector_db import ChromaVectorDB
from utils.database.relational_db.sqlite_db import SQLiteDB

class SemanticMemory:
    def __init__(self, db_configs):
        self.dbs = {
            'vector': ChromaVectorDB(
                #configs
            ),
            'sqlite': SQLiteDB(
                #configs
            )
        }
        self.retrieval_methods = {
            'reflection': ReflectionRetrieval(self.dbs['vector'], ReflectionTransform),
            'summary': SummaryRetrieval(self.dbs['sqlite'], SummaryTransform),
        }

```

### Databases
Each memory type can use multiple databases to hold data, where each database must have a way to count entries (see the base class `BaseDB` in [`base_db.py`](../utils/database/base_db.py))

To custom a db class, inherit from `BaseDB` and define the `retrieve` and `update` methods for that DB.

### Retrieval/Update classes
These are expressed as classes (eg `ReflectionRetrieval`, `SummaryRetrieval`) with `retrieve` or `update` methods to encapsulate the database details and handle optional transformations (eg `ReflectionTransform`, `SummaryTransform`).

### Retrieval/Update methods
The Memory class provides retrieval/update methods with human-readable, task-oriented names that utilize the retrieval/update classes.
```python
    def retrieve_reflections(self, query, **kwargs):
        return self.retrieval_methods['reflection'].retrieve(query, **kwargs)

    def retrieve_summaries(self, query, **kwargs):
        return self.retrieval_methods['summary'].retrieve(query, **kwargs)
```

We have both classes and methods so that customization with repeated elements (eg same database, similar parameters) is possible
## Example Usage

```python
# Instantiate the SemanticMemory class
semantic_memory = SemanticMemory(db_configs)

# Retrieve reflections
reflections = semantic_memory.retrieve_reflections(query="What are my recent reflections?")

# Retrieve summaries
summaries = semantic_memory.retrieve_summaries(query="Summarize my recent activities")
```

## More examples
We included [`BaseMem`](../memories/base_mem.py) as an example with a vector DB with retrieval and update methods and classes