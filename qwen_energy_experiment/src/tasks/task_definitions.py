"""
Task definitions for the Qwen energy consumption experiment.
Defines standardized prompts and task categories for systematic evaluation.

Implementation of Paper Section 4.1 (Subjects Selection) and Table 3 (Selected Inference Subjects).
Provides the four task categories described in Section 3.2.2 (RQ-2) for investigating
task complexity impact on energy consumption.
"""

import logging
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class TaskCategory(Enum):
    """
    Categories of inference tasks with different computational characteristics.
    
    Implements Paper Table 3 task categorization by complexity:
    - FACTUAL_LOOKUP: Low complexity (Paper: "What is the capital of France?")
    - TEXT_SUMMARIZATION: Medium complexity (Paper: "Summarize this article...")  
    - MULTI_STEP_REASONING: High complexity (Paper: "If a train travels...")
    - CODE_GENERATION: High complexity (Paper: "Write a Python code...")
    """
    
    FACTUAL_LOOKUP = "factual_lookup"
    TEXT_SUMMARIZATION = "text_summarization" 
    MULTI_STEP_REASONING = "multi_step_reasoning"
    CODE_GENERATION = "code_generation"


@dataclass
class TaskDefinition:
    """Definition of a specific task with prompts and configuration."""
    
    category: TaskCategory
    name: str
    description: str
    complexity: str  # "Low", "Medium", "High"
    prompts: List[str]
    expected_output_tokens: int
    max_tokens: int
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate task definition."""
        if not self.prompts:
            raise ValueError("Task must have at least one prompt")
        if len(self.prompts) < 20:
            logger.warning(f"Task {self.name} has only {len(self.prompts)} prompts, "
                          f"recommend 20 for statistical power")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")


class TaskLibrary:
    """Library of standardized tasks for the experiment."""
    
    def __init__(self):
        """Initialize task library with predefined tasks."""
        self.tasks = self._create_task_definitions()
        logger.info(f"Task library initialized with {len(self.tasks)} task types")
    
    def _create_task_definitions(self) -> Dict[str, TaskDefinition]:
        """Create all task definitions."""
        tasks = {}
        
        # Factual Lookup Tasks
        tasks['factual_simple'] = TaskDefinition(
            category=TaskCategory.FACTUAL_LOOKUP,
            name="factual_simple",
            description="Simple factual questions requiring direct knowledge retrieval",
            complexity="Low",
            prompts=self._create_factual_prompts(),
            expected_output_tokens=5,
            max_tokens=20,
            temperature=0.1,  # Low temperature for factual accuracy
            top_p=0.95
        )
        
        # Text Summarization Tasks
        tasks['summarization_short'] = TaskDefinition(
            category=TaskCategory.TEXT_SUMMARIZATION,
            name="summarization_short",
            description="Summarize short passages in 1-2 sentences",
            complexity="Medium",
            prompts=self._create_summarization_prompts(),
            expected_output_tokens=30,
            max_tokens=60,
            temperature=0.7,
            top_p=0.9
        )
        
        # Multi-step Reasoning Tasks
        tasks['reasoning_arithmetic'] = TaskDefinition(
            category=TaskCategory.MULTI_STEP_REASONING,
            name="reasoning_arithmetic",
            description="Multi-step arithmetic and logical reasoning",
            complexity="High",
            prompts=self._create_reasoning_prompts(),
            expected_output_tokens=50,
            max_tokens=100,
            temperature=0.3,  # Lower temperature for reasoning accuracy
            top_p=0.9
        )
        
        # Code Generation Tasks
        tasks['code_simple'] = TaskDefinition(
            category=TaskCategory.CODE_GENERATION,
            name="code_simple",
            description="Simple programming tasks and code snippets",
            complexity="High",
            prompts=self._create_code_prompts(),
            expected_output_tokens=80,
            max_tokens=150,
            temperature=0.2,  # Low temperature for code correctness
            top_p=0.95
        )
        
        return tasks
    
    def _create_factual_prompts(self) -> List[str]:
        """Create factual lookup prompts."""
        return [
            "What is the capital of France?",
            "Who wrote Romeo and Juliet?",
            "What is the largest planet in our solar system?",
            "In what year did World War II end?",
            "What is the chemical symbol for gold?",
            "Who painted the Mona Lisa?",
            "What is the smallest prime number?",
            "What is the speed of light in vacuum?",
            "Who invented the telephone?",
            "What is the currency of Japan?",
            "What is the highest mountain on Earth?",
            "Who wrote '1984'?",
            "What is the freezing point of water in Celsius?",
            "What is the largest ocean on Earth?",
            "Who discovered penicillin?",
            "What is the square root of 144?",
            "What is the most abundant gas in Earth's atmosphere?",
            "Who was the first person to walk on the moon?",
            "What is the atomic number of carbon?",
            "What is the capital of Australia?"
        ]
    
    def _create_summarization_prompts(self) -> List[str]:
        """Create text summarization prompts."""
        base_texts = [
            "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to natural intelligence displayed by animals and humans. Leading AI textbooks define the field as the study of 'intelligent agents': any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. Colloquially, the term 'artificial intelligence' is often used to describe machines that mimic cognitive functions that humans associate with the human mind, such as learning and problem-solving.",
            
            "Climate change refers to long-term shifts in global or regional climate patterns. Since the mid-20th century, scientists have observed unprecedented changes primarily attributed to increased levels of greenhouse gases produced by human activities, particularly the burning of fossil fuels. These changes manifest as rising global temperatures, melting ice caps, changing precipitation patterns, and more frequent extreme weather events.",
            
            "The Internet of Things (IoT) describes the network of physical objects that are embedded with sensors, software, and other technologies for the purpose of connecting and exchanging data with other devices and systems over the internet. These devices range from ordinary household objects to sophisticated industrial tools. With more than 7 billion connected IoT devices today, experts are expecting this number to grow to 10 billion by 2020 and 22 billion by 2025.",
            
            "Quantum computing is a type of computation that harnesses the collective properties of quantum states, such as superposition, interference, and entanglement, to process information. The devices that perform quantum computations are known as quantum computers. Though current quantum computers are too small to outperform usual computers for practical applications, they are believed to be capable of solving certain computational problems exponentially faster than classical computers."
        ]
        
        prompts = []
        for i, text in enumerate(base_texts):
            prompts.extend([
                f"Summarize this text in one sentence: {text}",
                f"Provide a brief summary of the following passage: {text}",
                f"In 1-2 sentences, explain what this text is about: {text}",
                f"Give a concise summary: {text}",
                f"What are the main points of this passage? {text}"
            ])
        
        return prompts[:20]  # Return exactly 20 prompts
    
    def _create_reasoning_prompts(self) -> List[str]:
        """Create multi-step reasoning prompts."""
        return [
            "If a train travels 60 km/h for 2 hours and then 80 km/h for 1.5 hours, what is the total distance traveled?",
            "A store sells apples for $2 per pound and oranges for $3 per pound. If you buy 4 pounds of apples and 2 pounds of oranges, how much do you pay?",
            "If all roses are flowers, and some flowers are red, can we conclude that some roses are red? Explain your reasoning.",
            "A rectangle has a length of 12 cm and a width of 8 cm. What is its perimeter and area?",
            "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
            "A number when divided by 6 gives remainder 4, and when divided by 4 gives remainder 2. What is the smallest such positive number?",
            "If today is Tuesday, what day will it be 100 days from now?",
            "A book costs $20. If the price is reduced by 25%, what is the new price?",
            "If 3 cats can catch 3 mice in 3 minutes, how many cats are needed to catch 100 mice in 100 minutes?",
            "A ladder 25 feet long is leaning against a wall. If the bottom of the ladder is 7 feet from the wall, how high up the wall does the ladder reach?",
            "If A = 1, B = 2, C = 3, ..., Z = 26, what is the sum of the letters in the word 'HELLO'?",
            "A tank is filled by two pipes. Pipe A fills it in 4 hours, Pipe B fills it in 6 hours. How long to fill it with both pipes?",
            "If you flip a fair coin 3 times, what is the probability of getting at least one head?",
            "A sequence starts: 2, 6, 12, 20, 30, ... What is the next number and what is the pattern?",
            "If 2^x = 32, what is the value of x?",
            "A car travels 240 miles using 8 gallons of gas. How many miles per gallon does it get?",
            "If the sum of two numbers is 50 and their difference is 10, what are the two numbers?",
            "How many different ways can you arrange the letters in the word 'CAT'?",
            "If a pizza is cut into 8 equal slices and you eat 3 slices, what fraction of the pizza is left?",
            "A clock shows 3:15. What is the angle between the hour and minute hands?"
        ]
    
    def _create_code_prompts(self) -> List[str]:
        """Create code generation prompts."""
        return [
            "Write a Python function to check if a number is prime.",
            "Create a function that reverses a string without using built-in methods.",
            "Write a Python function to find the factorial of a number.",
            "Create a function that checks if a string is a palindrome.",
            "Write code to find the maximum element in a list.",
            "Create a function that counts the number of vowels in a string.",
            "Write a Python function to calculate the Fibonacci sequence up to n terms.",
            "Create a function that removes duplicates from a list.",
            "Write code to check if two strings are anagrams.",
            "Create a function that finds the second largest number in a list.",
            "Write a Python function to convert Celsius to Fahrenheit.",
            "Create a function that counts words in a sentence.",
            "Write code to find the GCD of two numbers.",
            "Create a function that sorts a list of numbers without using sort().",
            "Write a Python function to check if a year is a leap year.",
            "Create a function that finds all even numbers in a list.",
            "Write code to calculate the area of a circle given the radius.",
            "Create a function that merges two sorted lists.",
            "Write a Python function to find the length of the longest word in a sentence.",
            "Create a function that generates a random password of specified length."
        ]
    
    def get_task(self, task_name: str) -> TaskDefinition:
        """Get a specific task definition."""
        if task_name not in self.tasks:
            available_tasks = list(self.tasks.keys())
            raise ValueError(f"Task '{task_name}' not found. Available: {available_tasks}")
        
        return self.tasks[task_name]
    
    def get_tasks_by_category(self, category: TaskCategory) -> List[TaskDefinition]:
        """Get all tasks for a specific category."""
        return [task for task in self.tasks.values() if task.category == category]
    
    def get_all_tasks(self) -> List[TaskDefinition]:
        """Get all task definitions."""
        return list(self.tasks.values())
    
    def get_task_summary(self) -> Dict[str, any]:
        """Get summary of all tasks."""
        summary = {
            'total_tasks': len(self.tasks),
            'tasks_by_category': {},
            'complexity_distribution': {'Low': 0, 'Medium': 0, 'High': 0}
        }
        
        for task in self.tasks.values():
            # Count by category
            category_name = task.category.value
            if category_name not in summary['tasks_by_category']:
                summary['tasks_by_category'][category_name] = 0
            summary['tasks_by_category'][category_name] += 1
            
            # Count by complexity
            if task.complexity in summary['complexity_distribution']:
                summary['complexity_distribution'][task.complexity] += 1
        
        return summary