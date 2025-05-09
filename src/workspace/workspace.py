from abc import ABC, abstractmethod
from pathlib import Path

from src.workspace.dummy_sink import DummySink
from src.workspace.wandb_sink import WandbSink


def _get_sink(workspace_config, user_config, agent_config, task_config):
    if workspace_config.sink == 'dummy':
        return DummySink()
    elif workspace_config.sink == 'wandb':
        return WandbSink(
            project=workspace_config.project,
            params=_get_params(user_config, agent_config, task_config, workspace_config),
            src_path=workspace_config.src_path)
    else:
        raise ValueError(f'Unknown sink implementation {workspace_config.sink}')


def _get_params(user_config, agent_config, task_config, workspace_config):
    return {
        'user.model': user_config.model,
        'agent.model': agent_config.model,
        'agent.agent': agent_config.agent,
        'agent.icl_count': agent_config.icl_count,
        'agent.num_ex_to_explore': agent_config.num_ex_to_explore,
        'task.task': task_config.task,
        'task.cost': task_config.cost,
        'task.num_train_ex': task_config.num_train_ex,
        'task.seed': task_config.seed,
        'is_dirty': workspace_config.is_dirty,
        'commit_id': workspace_config.commit_id 
    }


def get_last_part(s):
    return s.split("/")[-1] if "/" in s else s


class Workspace:
    def __init__(self, workspace_config, user_config, agent_config, task_config):
        self.sink = _get_sink(workspace_config, user_config, agent_config, task_config)
        
        # Debug printing for path construction
        print(f"workspace_config.log_folder: {workspace_config.log_folder}")
        print(f"user_config: {repr(user_config)}")
        print(f"agent_config: {repr(agent_config)}")
        print(f"task_config: {repr(task_config)}")
        
        _repr = '.'.join([
            get_last_part(repr(user_config)), "/",
            get_last_part(repr(agent_config)),
            get_last_part(repr(task_config))
        ])
        print(f"Constructed _repr: {_repr}")
        
        # Ensure the log folder exists
        log_folder_path = Path(workspace_config.log_folder)
        print(f"log_folder_path: {log_folder_path}")
        print(f"log_folder_path exists: {log_folder_path.exists()}")
        
        try:
            log_folder_path.mkdir(parents=True, exist_ok=True)
            print(f"Created log folder: {log_folder_path}")
        except Exception as e:
            print(f"Error creating log folder: {e}")
            raise
        
        self.log_path = log_folder_path / Path(workspace_config.log_filename or f'{_repr}.ndjson')
        print(f"Final log_path: {self.log_path}")
        print(f"log_path parent exists: {self.log_path.parent.exists()}")
        
        # Ensure parent directory exists
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Created parent directory: {self.log_path.parent}")
        except Exception as e:
            print(f"Error creating parent directory: {e}")
            raise
        
        self.conversation_log = []
        self.log_level = workspace_config.log_level
        if self.log_path.exists():
            self.log_path.unlink()

    def log(self, metrics):
        import json
        self.sink.log(metrics)
        try:
            # Ensure directory exists before writing
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, 'a') as f:
                f.write(f'{json.dumps(dict(metrics, conversation = self.conversation_log))}\n')
        except Exception as e:
            print(f"Error writing to log file: {e}")
            print(f"log_path: {self.log_path}")
            print(f"log_path parent exists: {self.log_path.parent.exists()}")
            raise
        self.conversation_log = []

    def log_message(self, request, llm_name, response):
        self.conversation_log.append({'q': request, llm_name: response})

    def stop(self):
        self.sink.log_artifacts('logs', 'logs', [self.log_path])
