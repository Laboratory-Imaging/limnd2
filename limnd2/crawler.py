from pathlib import Path
from typing import Union, Callable, Any
import os

from concurrent.futures import ProcessPoolExecutor

class FileCrawler:
    _folder: Path
    _file_extensions: Union[list[str], None] = None
    _recursive: bool = True

    def __init__(self, folder: Union[str, Path], 
                 file_extensions: Union[str, list[str], None] = None,
                 recursive: bool = False
                ):
        
        self.folder = folder
        self.file_extensions = file_extensions
        self.recursive = recursive
    
    @property
    def folder(self) -> Path:
        return self._folder
    
    @folder.setter
    def folder(self, folder: Union[str, Path]):
        if isinstance(folder, str):
            folder = Path(folder)

        self._folder = folder
    

    @property
    def file_extensions(self) -> list[str]:
        return self._file_extensions
    
    @file_extensions.setter
    def file_extensions(self, extensions: Union[str, list[str], None]):
        if extensions == None:
            self._file_extensions = None

        elif isinstance(extensions, str):
            if not extensions.startswith("."):
                extensions = "." + extensions
            self._file_extensions = [extensions.lower()]

        else:
            ext_lst = []
            for ext in extensions:
                if not ext.startswith("."):
                    ext = "." + ext 
                ext_lst.append(ext.lower())
            self._file_extensions = ext_lst


    @property
    def recursive(self) -> bool:
        return self._recursive
    
    @recursive.setter
    def recursive(self, recursive: bool):
        self._recursive = recursive


    def run(self, 
            function: Callable[[Path, ...], Any] = None,                # function to call for every path # type: ignore
            function_args: Union[dict[str, Any], None] = None,          # additional function arguments -> function(path, **function_args)
            use_concurrency: bool = False                               # use PoolExecutor
           ) -> Union[list[Path], dict[Path, Any]]:
        
        """
        Run file crawler.
        If function is not provided, this method will return list of found paths.
        If function is provided, it will return dictionary mapping each path to the result of the function call.
        Function is called like this: function(path, **function_args).
        If use_concurrency is set to true, function will be run with ProcessPoolExecutor() for each path.
        """

        if not function_args:
            function_args = {}

        if not self._folder.exists() or not self._folder.is_dir():
            raise ValueError("The provided folder path is not valid or is not a directory.")
        
        result = None

        if function:
            if use_concurrency:
                with ProcessPoolExecutor() as executor:
                    if self._recursive:
                        result = self._recursive_search_with_function(function, function_args, executor)
                    else:
                        result = self._non_recursive_search_with_function(function, function_args, executor)

                    # Collect results from futures
                    for path, future in result.items():
                        result[path] = future.result()
                        
            else:
                if self._recursive:
                    result =self._recursive_search_with_function(function, function_args)
                else:
                    result =self._non_recursive_search_with_function(function, function_args)
        else:
            if self._recursive:
                result = self._recursive_search_without_function() 
            else:
                result = self._non_recursive_search_without_function() 
        return result

        
    def _recursive_search_with_function(self, 
                                        function: Callable[[Path, ...], Any] = None,                # function to call for every path # type: ignore
                                        function_args: Union[dict[str, Any], None] = None,          # additional function arguments -> function(path, **function_args)
                                        executor: Union[ProcessPoolExecutor, None] = None
                                       ) -> dict[Path, Any]:
        
        result = {}
        for root, _, files in os.walk(self._folder):
            for file in files:
                file_path = Path(root) / file
                if self._file_extensions and file_path.suffix.lower() not in self._file_extensions:
                    continue

                if executor:
                    result[file_path] = executor.submit(function, file_path, **function_args)
                else:
                    result[file_path] = function(file_path, **function_args)
        
        return result
        
    def _non_recursive_search_with_function(self,
                                            function: Callable[[Path, ...], Any] = None,                # function to call for every path # type: ignore
                                            function_args: Union[dict[str, Any], None] = None,          # additional function arguments -> function(path, **function_args)  
                                            executor: Union[ProcessPoolExecutor, None] = None
                                           ) -> dict[Path, Any]:
        
        result = {}
        for file_path in self._folder.glob('*'):
            if file_path.is_file():
                if self._file_extensions and file_path.suffix.lower() not in self._file_extensions:
                    continue
                
                if executor:
                    result[file_path] = executor.submit(function, file_path, **function_args)
                else:
                    result[file_path] = function(file_path, **function_args)
        return result
    

    def _recursive_search_without_function(self) -> list[Path]:
        paths = []
        for root, _, files in os.walk(self._folder):
            for file in files:
                path = Path(root) / file
                if self._file_extensions and path.suffix.lower() not in self._file_extensions:
                    continue
                paths.append(path)
        return paths
        
    def _non_recursive_search_without_function(self) -> list[Path]:
        paths = []
        for path in self._folder.glob('*'):
            if path.is_file():
                if not self._file_extensions:
                    paths.append(path)    
                elif path.suffix.lower() in self._file_extensions:
                    paths.append(path)
        return paths
                