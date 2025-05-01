import json
import os
import numpy as np
import sympy as sp
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional

class Function:
    """
    Clase para representar una función matemática.
    Permite la evaluación y derivación de funciones.
    """
    def __init__(self, expression_str: str, variables: List[str]):
        """
        Inicializa una función con su expresión y variables.
        
        Args:
            expression_str: Expresión matemática como string (ej: "x**2 + y**2")
            variables: Lista de nombres de variables (ej: ["x", "y"])
        """
        self.expression_str = expression_str
        self.variables = variables
        
        # Convertir variables a símbolos de sympy
        self.symbols = [sp.Symbol(var) for var in variables]
        
        # Parsear la expresión
        try:
            self.expression = sp.sympify(expression_str)
        except Exception as e:
            raise ValueError(f"Error al parsear la función: {e}")
        
        # Calcular derivadas parciales
        self.derivatives = [sp.diff(self.expression, sym) for sym in self.symbols]
        
        # Calcular matriz Hessiana (segundas derivadas)
        self.hessian = [[sp.diff(self.derivatives[i], self.symbols[j]) 
                         for j in range(len(self.symbols))] 
                         for i in range(len(self.symbols))]
    
    def evaluate(self, point: List[float]) -> float:
        """
        Evalúa la función en un punto dado.
        
        Args:
            point: Lista de valores para las variables
            
        Returns:
            Valor de la función en el punto
        """
        if len(point) != len(self.variables):
            raise ValueError(f"El punto debe tener {len(self.variables)} dimensiones")
        
        # Crear diccionario de sustitución
        subs_dict = {self.symbols[i]: point[i] for i in range(len(self.symbols))}
        
        # Evaluar la expresión
        return float(self.expression.subs(subs_dict))
    
    def evaluate_gradient(self, point: List[float]) -> List[float]:
        """
        Evalúa el gradiente de la función en un punto dado.
        
        Args:
            point: Lista de valores para las variables
            
        Returns:
            Lista con los valores de las derivadas parciales en el punto
        """
        if len(point) != len(self.variables):
            raise ValueError(f"El punto debe tener {len(self.variables)} dimensiones")
        
        # Crear diccionario de sustitución
        subs_dict = {self.symbols[i]: point[i] for i in range(len(self.symbols))}
        
        # Evaluar cada derivada parcial
        return [float(deriv.subs(subs_dict)) for deriv in self.derivatives]
    
    def evaluate_hessian(self, point: List[float]) -> List[List[float]]:
        """
        Evalúa la matriz Hessiana de la función en un punto dado.
        
        Args:
            point: Lista de valores para las variables
            
        Returns:
            Matriz con los valores de las segundas derivadas en el punto
        """
        if len(point) != len(self.variables):
            raise ValueError(f"El punto debe tener {len(self.variables)} dimensiones")
        
        # Crear diccionario de sustitución
        subs_dict = {self.symbols[i]: point[i] for i in range(len(self.symbols))}
        
        # Evaluar cada elemento de la matriz Hessiana
        return [[float(self.hessian[i][j].subs(subs_dict)) 
                for j in range(len(self.symbols))] 
                for i in range(len(self.symbols))]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convierte la función a un diccionario para almacenamiento JSON.
        
        Returns:
            Diccionario con los datos de la función
        """
        return {
            "expression": self.expression_str,
            "variables": self.variables
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Function':
        """
        Crea una función a partir de un diccionario.
        
        Args:
            data: Diccionario con los datos de la función
            
        Returns:
            Objeto Function
        """
        return cls(data["expression"], data["variables"])
    
    def __str__(self) -> str:
        """Representación en string de la función"""
        return f"f({', '.join(self.variables)}) = {self.expression_str}"


class Constraint:
    """
    Clase para representar una restricción en un problema de optimización.
    Puede ser de igualdad (tipo "=") o desigualdad (tipo "<=", ">=").
    """
    def __init__(self, expression_str: str, variables: List[str], constraint_type: str = "="):
        """
        Inicializa una restricción.
        
        Args:
            expression_str: Expresión matemática como string (ej: "x + y - 1")
            variables: Lista de nombres de variables (ej: ["x", "y"])
            constraint_type: Tipo de restricción ("=", "<=", ">=")
        """
        self.expression_str = expression_str
        self.variables = variables
        
        # Validar tipo de restricción
        if constraint_type not in ["=", "<=", ">="]:
            raise ValueError("El tipo de restricción debe ser '=', '<=' o '>='")
        
        self.constraint_type = constraint_type
        
        # Convertir variables a símbolos de sympy
        self.symbols = [sp.Symbol(var) for var in variables]
        
        # Parsear la expresión
        try:
            self.expression = sp.sympify(expression_str)
        except Exception as e:
            raise ValueError(f"Error al parsear la restricción: {e}")
        
        # Calcular derivadas parciales
        self.derivatives = [sp.diff(self.expression, sym) for sym in self.symbols]
    
    def evaluate(self, point: List[float]) -> float:
        """
        Evalúa la restricción en un punto dado.
        
        Args:
            point: Lista de valores para las variables
            
        Returns:
            Valor de la restricción en el punto
        """
        if len(point) != len(self.variables):
            raise ValueError(f"El punto debe tener {len(self.variables)} dimensiones")
        
        # Crear diccionario de sustitución
        subs_dict = {self.symbols[i]: point[i] for i in range(len(self.symbols))}
        
        # Evaluar la expresión
        return float(self.expression.subs(subs_dict))
    
    def evaluate_gradient(self, point: List[float]) -> List[float]:
        """
        Evalúa el gradiente de la restricción en un punto dado.
        
        Args:
            point: Lista de valores para las variables
            
        Returns:
            Lista con los valores de las derivadas parciales en el punto
        """
        if len(point) != len(self.variables):
            raise ValueError(f"El punto debe tener {len(self.variables)} dimensiones")
        
        # Crear diccionario de sustitución
        subs_dict = {self.symbols[i]: point[i] for i in range(len(self.symbols))}
        
        # Evaluar cada derivada parcial
        return [float(deriv.subs(subs_dict)) for deriv in self.derivatives]
    
    def is_satisfied(self, point: List[float], tolerance: float = 1e-6) -> bool:
        """
        Verifica si un punto satisface la restricción.
        
        Args:
            point: Lista de valores para las variables
            tolerance: Tolerancia para restricciones de igualdad
            
        Returns:
            True si el punto satisface la restricción, False en caso contrario
        """
        value = self.evaluate(point)
        
        if self.constraint_type == "=":
            return abs(value) <= tolerance
        elif self.constraint_type == "<=":
            return value <= tolerance
        elif self.constraint_type == ">=":
            return value >= -tolerance
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convierte la restricción a un diccionario para almacenamiento JSON.
        
        Returns:
            Diccionario con los datos de la restricción
        """
        return {
            "expression": self.expression_str,
            "variables": self.variables,
            "constraint_type": self.constraint_type
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Constraint':
        """
        Crea una restricción a partir de un diccionario.
        
        Args:
            data: Diccionario con los datos de la restricción
            
        Returns:
            Objeto Constraint
        """
        return cls(data["expression"], data["variables"], data["constraint_type"])
    
    def __str__(self) -> str:
        """Representación en string de la restricción"""
        return f"g({', '.join(self.variables)}) {self.constraint_type} 0, donde g = {self.expression_str}"


class OptimizationProblem:
    """
    Clase para representar un problema de optimización completo.
    Incluye función objetivo y restricciones.
    """
    def __init__(self, 
                 objective_function: Function, 
                 constraints: List[Constraint] = None,
                 problem_type: str = "min",
                 name: str = "Problema sin nombre"):
        """
        Inicializa un problema de optimización.
        
        Args:
            objective_function: Función objetivo
            constraints: Lista de restricciones (opcional)
            problem_type: Tipo de problema ("min" o "max")
            name: Nombre descriptivo del problema
        """
        self.objective_function = objective_function
        self.constraints = constraints or []
        
        # Validar tipo de problema
        if problem_type not in ["min", "max"]:
            raise ValueError("El tipo de problema debe ser 'min' o 'max'")
        
        self.problem_type = problem_type
        self.name = name
        
        # Verificar que todas las restricciones usen las mismas variables
        for constraint in self.constraints:
            if set(constraint.variables) != set(objective_function.variables):
                raise ValueError("Todas las restricciones deben usar las mismas variables que la función objetivo")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convierte el problema a un diccionario para almacenamiento JSON.
        
        Returns:
            Diccionario con los datos del problema
        """
        return {
            "name": self.name,
            "problem_type": self.problem_type,
            "objective_function": self.objective_function.to_dict(),
            "constraints": [constraint.to_dict() for constraint in self.constraints],
            "variables": self.objective_function.variables
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationProblem':
        """
        Crea un problema a partir de un diccionario.
        
        Args:
            data: Diccionario con los datos del problema
            
        Returns:
            Objeto OptimizationProblem
        """
        objective_function = Function.from_dict(data["objective_function"])
        constraints = [Constraint.from_dict(constraint_data) for constraint_data in data["constraints"]]
        
        return cls(
            objective_function=objective_function,
            constraints=constraints,
            problem_type=data["problem_type"],
            name=data["name"]
        )
    
    def __str__(self) -> str:
        """Representación en string del problema"""
        problem_str = f"Problema: {self.name}\n"
        problem_str += f"Tipo: {self.problem_type}imizar\n"
        problem_str += f"Función objetivo: {self.objective_function}\n"
        
        if self.constraints:
            problem_str += "Restricciones:\n"
            for i, constraint in enumerate(self.constraints, 1):
                problem_str += f"  {i}. {constraint}\n"
        else:
            problem_str += "Sin restricciones\n"
        
        return problem_str


class OptimizationResult:
    """
    Clase para almacenar y mostrar los resultados de un problema de optimización.
    """
    def __init__(self, 
                 problem: OptimizationProblem,
                 optimal_point: List[float],
                 optimal_value: float,
                 method_name: str,
                 iterations: int = None,
                 convergence_history: List[float] = None,
                 lagrange_multipliers: List[float] = None,
                 is_feasible: bool = True,
                 message: str = "Optimización exitosa"):
        """
        Inicializa un resultado de optimización.
        
        Args:
            problem: Problema de optimización resuelto
            optimal_point: Punto óptimo encontrado
            optimal_value: Valor de la función objetivo en el punto óptimo
            method_name: Nombre del método utilizado
            iterations: Número de iteraciones realizadas (opcional)
            convergence_history: Historial de valores de la función objetivo (opcional)
            lagrange_multipliers: Multiplicadores de Lagrange (opcional)
            is_feasible: Indica si la solución es factible
            message: Mensaje descriptivo del resultado
        """
        self.problem = problem
        self.optimal_point = optimal_point
        self.optimal_value = optimal_value
        self.method_name = method_name
        self.iterations = iterations
        self.convergence_history = convergence_history or []
        self.lagrange_multipliers = lagrange_multipliers
        self.is_feasible = is_feasible
        self.message = message
        
        # Verificar factibilidad con las restricciones
        if problem.constraints and is_feasible:
            self.is_feasible = all(constraint.is_satisfied(optimal_point) 
                                  for constraint in problem.constraints)
    
    def plot_convergence(self):
        """
        Genera un gráfico de convergencia si hay historial disponible.
        """
        if not self.convergence_history:
            print("No hay historial de convergencia disponible para graficar.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.convergence_history)), self.convergence_history, 'b-', linewidth=2)
        plt.title(f'Convergencia del método {self.method_name}')
        plt.xlabel('Iteraciones')
        plt.ylabel('Valor de la función objetivo')
        plt.grid(True)
        plt.show()
    
    def __str__(self) -> str:
        """Representación en string del resultado"""
        variables = self.problem.objective_function.variables
        
        result_str = f"\n{'='*50}\n"
        result_str += f"RESULTADOS DE OPTIMIZACIÓN\n"
        result_str += f"{'='*50}\n\n"
        
        result_str += f"Método utilizado: {self.method_name}\n\n"
        
        result_str += "Punto óptimo encontrado:\n"
        for i, var in enumerate(variables):
            result_str += f"  {var} = {self.optimal_point[i]:.6f}\n"
        
        result_str += f"\nValor óptimo: {self.optimal_value:.6f}\n"
        
        if self.lagrange_multipliers:
            result_str += "\nMultiplicadores de Lagrange:\n"
            for i, multiplier in enumerate(self.lagrange_multipliers):
                result_str += f"  λ{i+1} = {multiplier:.6f}\n"
        
        if self.iterations is not None:
            result_str += f"\nNúmero de iteraciones: {self.iterations}\n"
        
        result_str += f"\nFactibilidad: {'Solución factible' if self.is_feasible else 'Solución no factible'}\n"
        result_str += f"Mensaje: {self.message}\n"
        
        return result_str


class OptimizationMethod(ABC):
    """
    Clase abstracta base para todos los métodos de optimización.
    Define la interfaz común que deben implementar todos los métodos.
    """
    @abstractmethod
    def solve(self, problem: OptimizationProblem, initial_point: List[float] = None, **kwargs) -> OptimizationResult:
        """
        Resuelve un problema de optimización.
        
        Args:
            problem: Problema de optimización a resolver
            initial_point: Punto inicial para métodos iterativos (opcional)
            **kwargs: Argumentos adicionales específicos del método
            
        Returns:
            Resultado de la optimización
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Retorna el nombre del método.
        
        Returns:
            Nombre del método
        """
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """
        Retorna una descripción del método.
        
        Returns:
            Descripción del método
        """
        pass


class UnconstrainedOptimization(OptimizationMethod):
    """
    Método para optimización sin restricciones usando derivadas parciales.
    Encuentra puntos críticos y los clasifica usando la matriz Hessiana.
    """
    def solve(self, problem: OptimizationProblem, initial_point: List[float] = None, **kwargs) -> OptimizationResult:
        """
        Resuelve un problema de optimización sin restricciones.
        
        Args:
            problem: Problema de optimización a resolver
            initial_point: No utilizado en este método
            **kwargs: Argumentos adicionales
            
        Returns:
            Resultado de la optimización
        """
        # Verificar que no haya restricciones
        if problem.constraints:
            return OptimizationResult(
                problem=problem,
                optimal_point=[0] * len(problem.objective_function.variables),
                optimal_value=0,
                method_name=self.get_name(),
                is_feasible=False,
                message="Este método solo funciona para problemas sin restricciones"
            )
        
        # Obtener símbolos y variables
        variables = problem.objective_function.variables
        symbols = problem.objective_function.symbols
        
        # Obtener expresión y derivadas
        expr = problem.objective_function.expression
        derivatives = [sp.diff(expr, sym) for sym in symbols]
        
        # Crear sistema de ecuaciones igualando derivadas a cero
        equations = derivatives
        
        try:
            # Resolver sistema de ecuaciones
            solutions = sp.solve(equations, symbols, dict=True)
            
            if not solutions:
                return OptimizationResult(
                    problem=problem,
                    optimal_point=[0] * len(variables),
                    optimal_value=0,
                    method_name=self.get_name(),
                    is_feasible=False,
                    message="No se encontraron puntos críticos"
                )
            
            # Evaluar cada solución
            results = []
            for solution in solutions:
                # Convertir solución a lista
                point = [float(solution.get(sym, 0)) for sym in symbols]
                
                # Evaluar función objetivo
                value = problem.objective_function.evaluate(point)
                
                # Evaluar matriz Hessiana
                hessian = problem.objective_function.evaluate_hessian(point)
                hessian_matrix = np.array(hessian)
                
                # Determinar tipo de punto crítico
                try:
                    eigenvalues = np.linalg.eigvals(hessian_matrix)
                    
                    if all(eigenvalues > 0):
                        point_type = "mínimo"
                    elif all(eigenvalues < 0):
                        point_type = "máximo"
                    else:
                        point_type = "punto de silla"
                except:
                    point_type = "indeterminado"
                
                # Guardar resultado
                results.append({
                    "point": point,
                    "value": value,
                    "type": point_type
                })
            
            # Seleccionar mejor solución según tipo de problema
            if problem.problem_type == "min":
                best_result = min(results, key=lambda x: x["value"])
            else:
                best_result = max(results, key=lambda x: x["value"])
            
            return OptimizationResult(
                problem=problem,
                optimal_point=best_result["point"],
                optimal_value=best_result["value"],
                method_name=self.get_name(),
                message=f"Punto crítico encontrado: {best_result['type']}"
            )
            
        except Exception as e:
            return OptimizationResult(
                problem=problem,
                optimal_point=[0] * len(variables),
                optimal_value=0,
                method_name=self.get_name(),
                is_feasible=False,
                message=f"Error al resolver el sistema de ecuaciones: {str(e)}"
            )
    
    def get_name(self) -> str:
        return "Optimización sin Restricciones (Derivadas Parciales)"
    
    def get_description(self) -> str:
        return ("Este método encuentra puntos críticos de una función sin restricciones "
                "igualando todas las derivadas parciales a cero y clasificando los puntos "
                "usando la matriz Hessiana.")


class GradientDescent(OptimizationMethod):
    """
    Método del Gradiente Descendente para optimización sin restricciones.
    """
    def solve(self, problem: OptimizationProblem, initial_point: List[float] = None, **kwargs) -> OptimizationResult:
        """
        Resuelve un problema de optimización usando el método del gradiente descendente.
        
        Args:
            problem: Problema de optimización a resolver
            initial_point: Punto inicial para el método
            **kwargs: Argumentos adicionales
                max_iterations: Número máximo de iteraciones
                learning_rate: Tasa de aprendizaje
                tolerance: Tolerancia para convergencia
            
        Returns:
            Resultado de la optimización
        """
        # Parámetros del método
        max_iterations = kwargs.get("max_iterations", 1000)
        learning_rate = kwargs.get("learning_rate", 0.01)
        tolerance = kwargs.get("tolerance", 1e-6)
        
        # Verificar punto inicial
        variables = problem.objective_function.variables
        if initial_point is None:
            initial_point = [0.0] * len(variables)
        
        if len(initial_point) != len(variables):
            return OptimizationResult(
                problem=problem,
                optimal_point=[0] * len(variables),
                optimal_value=0,
                method_name=self.get_name(),
                is_feasible=False,
                message=f"El punto inicial debe tener {len(variables)} dimensiones"
            )
        
        # Ajustar dirección según tipo de problema
        direction = 1 if problem.problem_type == "min" else -1
        
        # Inicializar variables
        current_point = initial_point.copy()
        iterations = 0
        convergence_history = []
        
        # Iterar hasta convergencia o máximo de iteraciones
        while iterations < max_iterations:
            # Evaluar función objetivo
            current_value = problem.objective_function.evaluate(current_point)
            convergence_history.append(current_value)
            
            # Calcular gradiente
            gradient = problem.objective_function.evaluate_gradient(current_point)
            
            # Verificar magnitud del gradiente para convergencia
            if np.linalg.norm(gradient) < tolerance:
                break
            
            # Actualizar punto
            new_point = [current_point[i] - direction * learning_rate * gradient[i] 
                         for i in range(len(current_point))]
            
            # Verificar cambio en el punto para convergencia
            if np.linalg.norm(np.array(new_point) - np.array(current_point)) < tolerance:
                break
            
            current_point = new_point
            iterations += 1
        
        # Evaluar función objetivo en el punto final
        optimal_value = problem.objective_function.evaluate(current_point)
        
        return OptimizationResult(
            problem=problem,
            optimal_point=current_point,
            optimal_value=optimal_value,
            method_name=self.get_name(),
            iterations=iterations,
            convergence_history=convergence_history,
            message=f"Convergencia alcanzada en {iterations} iteraciones"
        )
    
    def get_name(self) -> str:
        return "Método del Gradiente Descendente"
    
    def get_description(self) -> str:
        return ("Este método iterativo minimiza una función siguiendo la dirección "
                "del gradiente negativo. Es efectivo para funciones convexas sin "
                "restricciones.")


class ProjectedGradient(OptimizationMethod):
    """
    Método del Gradiente Proyectado para optimización con restricciones de desigualdad.
    """
    def solve(self, problem: OptimizationProblem, initial_point: List[float] = None, **kwargs) -> OptimizationResult:
        """
        Resuelve un problema de optimización usando el método del gradiente proyectado.
        
        Args:
            problem: Problema de optimización a resolver
            initial_point: Punto inicial para el método
            **kwargs: Argumentos adicionales
                max_iterations: Número máximo de iteraciones
                learning_rate: Tasa de aprendizaje
                tolerance: Tolerancia para convergencia
            
        Returns:
            Resultado de la optimización
        """
        # Parámetros del método
        max_iterations = kwargs.get("max_iterations", 1000)
        learning_rate = kwargs.get("learning_rate", 0.01)
        tolerance = kwargs.get("tolerance", 1e-6)
        
        # Verificar punto inicial
        variables = problem.objective_function.variables
        if initial_point is None:
            initial_point = [0.0] * len(variables)
        
        if len(initial_point) != len(variables):
            return OptimizationResult(
                problem=problem,
                optimal_point=[0] * len(variables),
                optimal_value=0,
                method_name=self.get_name(),
                is_feasible=False,
                message=f"El punto inicial debe tener {len(variables)} dimensiones"
            )
        
        # Verificar factibilidad del punto inicial
        is_feasible = all(constraint.is_satisfied(initial_point) for constraint in problem.constraints)
        if not is_feasible:
            return OptimizationResult(
                problem=problem,
                optimal_point=initial_point,
                optimal_value=problem.objective_function.evaluate(initial_point),
                method_name=self.get_name(),
                is_feasible=False,
                message="El punto inicial no es factible"
            )
        
        # Ajustar dirección según tipo de problema
        direction = 1 if problem.problem_type == "min" else -1
        
        # Inicializar variables
        current_point = initial_point.copy()
        iterations = 0
        convergence_history = []
        
        # Iterar hasta convergencia o máximo de iteraciones
        while iterations < max_iterations:
            # Evaluar función objetivo
            current_value = problem.objective_function.evaluate(current_point)
            convergence_history.append(current_value)
            
            # Calcular gradiente
            gradient = problem.objective_function.evaluate_gradient(current_point)
            
            # Verificar magnitud del gradiente para convergencia
            if np.linalg.norm(gradient) < tolerance:
                break
            
            # Calcular nuevo punto sin proyección
            new_point = [current_point[i] - direction * learning_rate * gradient[i] 
                         for i in range(len(current_point))]
            
            # Proyectar el punto a la región factible
            # Implementación simple: si el punto no es factible, reducir el paso
            step_size = learning_rate
            while step_size > tolerance:
                new_point = [current_point[i] - direction * step_size * gradient[i] 
                             for i in range(len(current_point))]
                
                # Verificar factibilidad
                is_feasible = all(constraint.is_satisfied(new_point) for constraint in problem.constraints)
                if is_feasible:
                    break
                
                # Reducir paso
                step_size *= 0.5
            
            # Si no se encontró un punto factible, terminar
            if not is_feasible:
                break
            
            # Verificar cambio en el punto para convergencia
            if np.linalg.norm(np.array(new_point) - np.array(current_point)) < tolerance:
                break
            
            current_point = new_point
            iterations += 1
        
        # Evaluar función objetivo en el punto final
        optimal_value = problem.objective_function.evaluate(current_point)
        
        return OptimizationResult(
            problem=problem,
            optimal_point=current_point,
            optimal_value=optimal_value,
            method_name=self.get_name(),
            iterations=iterations,
            convergence_history=convergence_history,
            is_feasible=is_feasible,
            message=f"Convergencia alcanzada en {iterations} iteraciones"
        )
    
    def get_name(self) -> str:
        return "Método del Gradiente Proyectado"
    
    def get_description(self) -> str:
        return ("Este método extiende el gradiente descendente para problemas con restricciones "
                "de desigualdad, proyectando cada paso a la región factible.")


class LagrangeMultipliers(OptimizationMethod):
    """
    Método de Multiplicadores de Lagrange para optimización con restricciones de igualdad.
    """
    def solve(self, problem: OptimizationProblem, initial_point: List[float] = None, **kwargs) -> OptimizationResult:
        """
        Resuelve un problema de optimización usando multiplicadores de Lagrange.
        
        Args:
            problem: Problema de optimización a resolver
            initial_point: No utilizado en este método
            **kwargs: Argumentos adicionales
            
        Returns:
            Resultado de la optimización
        """
        # Verificar que solo haya restricciones de igualdad
        equality_constraints = [c for c in problem.constraints if c.constraint_type == "="]
        if len(equality_constraints) != len(problem.constraints):
            return OptimizationResult(
                problem=problem,
                optimal_point=[0] * len(problem.objective_function.variables),
                optimal_value=0,
                method_name=self.get_name(),
                is_feasible=False,
                message="Este método solo funciona con restricciones de igualdad"
            )
        
        # Obtener símbolos y variables
        variables = problem.objective_function.variables
        symbols = problem.objective_function.symbols
        
        # Crear símbolos para multiplicadores de Lagrange
        lambda_symbols = [sp.Symbol(f"lambda_{i}") for i in range(len(equality_constraints))]
        
        # Construir función de Lagrange
        lagrangian = problem.objective_function.expression
        
        for i, constraint in enumerate(equality_constraints):
            lagrangian += lambda_symbols[i] * constraint.expression
        
        # Calcular derivadas parciales respecto a variables y multiplicadores
        derivatives_vars = [sp.diff(lagrangian, sym) for sym in symbols]
        derivatives_lambda = [constraint.expression for constraint in equality_constraints]
        
        # Crear sistema de ecuaciones
        equations = derivatives_vars + derivatives_lambda
        all_symbols = symbols + lambda_symbols
        
        try:
            # Resolver sistema de ecuaciones
            solutions = sp.solve(equations, all_symbols, dict=True)
            
            if not solutions:
                return OptimizationResult(
                    problem=problem,
                    optimal_point=[0] * len(variables),
                    optimal_value=0,
                    method_name=self.get_name(),
                    is_feasible=False,
                    message="No se encontraron puntos críticos"
                )
            
            # Evaluar cada solución
            results = []
            for solution in solutions:
                # Verificar que la solución sea real
                if not all(sp.im(solution.get(sym, 0)) == 0 for sym in symbols):
                    continue
                
                # Convertir solución a lista
                point = [float(solution.get(sym, 0)) for sym in symbols]
                multipliers = [float(solution.get(lam, 0)) for lam in lambda_symbols]
                
                # Evaluar función objetivo
                value = problem.objective_function.evaluate(point)
                
                # Evaluar matriz Hessiana para clasificar punto
                hessian = problem.objective_function.evaluate_hessian(point)
                
                # Guardar resultado
                results.append({
                    "point": point,
                    "value": value,
                    "multipliers": multipliers
                })
            
            if not results:
                return OptimizationResult(
                    problem=problem,
                    optimal_point=[0] * len(variables),
                    optimal_value=0,
                    method_name=self.get_name(),
                    is_feasible=False,
                    message="No se encontraron soluciones reales"
                )
            
            # Seleccionar mejor solución según tipo de problema
            if problem.problem_type == "min":
                best_result = min(results, key=lambda x: x["value"])
            else:
                best_result = max(results, key=lambda x: x["value"])
            
            return OptimizationResult(
                problem=problem,
                optimal_point=best_result["point"],
                optimal_value=best_result["value"],
                method_name=self.get_name(),
                lagrange_multipliers=best_result["multipliers"],
                message="Punto crítico encontrado con multiplicadores de Lagrange"
            )
            
        except Exception as e:
            return OptimizationResult(
                problem=problem,
                optimal_point=[0] * len(variables),
                optimal_value=0,
                method_name=self.get_name(),
                is_feasible=False,
                message=f"Error al resolver el sistema de ecuaciones: {str(e)}"
            )
    
    def get_name(self) -> str:
        return "Método de Multiplicadores de Lagrange"
    
    def get_description(self) -> str:
        return ("Este método encuentra puntos críticos de una función con restricciones "
                "de igualdad mediante la construcción de la función de Lagrange.")


class KKTMethod(OptimizationMethod):
    """
    Método de Karush-Kuhn-Tucker (KKT) para optimización con restricciones de desigualdad.
    """
    def solve(self, problem: OptimizationProblem, initial_point: List[float] = None, **kwargs) -> OptimizationResult:
        """
        Resuelve un problema de optimización usando las condiciones KKT.
        
        Args:
            problem: Problema de optimización a resolver
            initial_point: No utilizado en este método
            **kwargs: Argumentos adicionales
            
        Returns:
            Resultado de la optimización
        """
        # Obtener símbolos y variables
        variables = problem.objective_function.variables
        symbols = problem.objective_function.symbols
        
        # Separar restricciones por tipo
        equality_constraints = [c for c in problem.constraints if c.constraint_type == "="]
        inequality_constraints = [c for c in problem.constraints if c.constraint_type in ["<=", ">="]]
        
        # Normalizar restricciones de desigualdad a forma <= 0
        normalized_inequalities = []
        for constraint in inequality_constraints:
            if constraint.constraint_type == ">=":
                # Convertir g(x) >= 0 a -g(x) <= 0
                expr_str = f"-({constraint.expression_str})"
                normalized_inequalities.append(Constraint(expr_str, variables, "<="))
            else:
                normalized_inequalities.append(constraint)
        
        # Crear símbolos para multiplicadores
        lambda_symbols = [sp.Symbol(f"lambda_{i}") for i in range(len(equality_constraints))]
        mu_symbols = [sp.Symbol(f"mu_{i}", positive=True) for i in range(len(normalized_inequalities))]
        
        # Construir función Lagrangiana
        lagrangian = problem.objective_function.expression
        
        # Ajustar signo según tipo de problema
        sign = 1 if problem.problem_type == "min" else -1
        lagrangian *= sign
        
        # Agregar términos de restricciones
        for i, constraint in enumerate(equality_constraints):
            lagrangian += lambda_symbols[i] * constraint.expression
        
        for i, constraint in enumerate(normalized_inequalities):
            lagrangian += mu_symbols[i] * constraint.expression
        
        # Calcular derivadas parciales respecto a variables
        derivatives_vars = [sp.diff(lagrangian, sym) for sym in symbols]
        
        # Condiciones KKT
        # 1. Estacionariedad: ∇L = 0
        kkt_stationarity = derivatives_vars
        
        # 2. Factibilidad primal
        kkt_primal_feasibility_eq = [constraint.expression for constraint in equality_constraints]
        kkt_primal_feasibility_ineq = [constraint.expression for constraint in normalized_inequalities]
        
        # 3. Complementariedad: μᵢ·gᵢ(x) = 0
        kkt_complementary_slackness = [mu_symbols[i] * normalized_inequalities[i].expression 
                                      for i in range(len(normalized_inequalities))]
        
        # 4. Factibilidad dual: μᵢ ≥ 0 (ya impuesto en la definición de símbolos)
        
        # Crear sistema de ecuaciones
        equations = (kkt_stationarity + kkt_primal_feasibility_eq + 
                    kkt_complementary_slackness)
        
        # Agregar restricciones de desigualdad como condiciones
        conditions = []
        for expr in kkt_primal_feasibility_ineq:
            conditions.append(expr <= 0)
        
        all_symbols = symbols + lambda_symbols + mu_symbols
        
        try:
            # Resolver sistema de ecuaciones
            # Nota: sp.solve no maneja bien las condiciones de desigualdad en sistemas no lineales
            # Esta es una simplificación que puede no encontrar todas las soluciones
            solutions = sp.solve(equations, all_symbols, dict=True)
            
            if not solutions:
                return OptimizationResult(
                    problem=problem,
                    optimal_point=[0] * len(variables),
                    optimal_value=0,
                    method_name=self.get_name(),
                    is_feasible=False,
                    message="No se encontraron puntos críticos"
                )
            
            # Evaluar cada solución
            results = []
            for solution in solutions:
                # Verificar que la solución sea real
                if not all(sp.im(solution.get(sym, 0)) == 0 for sym in symbols):
                    continue
                
                # Convertir solución a lista
                point = [float(solution.get(sym, 0)) for sym in symbols]
                
                # Verificar factibilidad
                is_feasible = True
                for constraint in problem.constraints:
                    if not constraint.is_satisfied(point):
                        is_feasible = False
                        break
                
                if not is_feasible:
                    continue
                
                # Evaluar función objetivo
                value = problem.objective_function.evaluate(point)
                
                # Obtener multiplicadores
                lambda_values = [float(solution.get(lam, 0)) for lam in lambda_symbols]
                mu_values = [float(solution.get(mu, 0)) for mu in mu_symbols]
                multipliers = lambda_values + mu_values
                
                # Guardar resultado
                results.append({
                    "point": point,
                    "value": value,
                    "multipliers": multipliers,
                    "is_feasible": is_feasible
                })
            
            if not results:
                return OptimizationResult(
                    problem=problem,
                    optimal_point=[0] * len(variables),
                    optimal_value=0,
                    method_name=self.get_name(),
                    is_feasible=False,
                    message="No se encontraron soluciones factibles"
                )
            
            # Seleccionar mejor solución según tipo de problema
            if problem.problem_type == "min":
                best_result = min(results, key=lambda x: x["value"])
            else:
                best_result = max(results, key=lambda x: x["value"])
            
            return OptimizationResult(
                problem=problem,
                optimal_point=best_result["point"],
                optimal_value=best_result["value"],
                method_name=self.get_name(),
                lagrange_multipliers=best_result["multipliers"],
                is_feasible=best_result["is_feasible"],
                message="Punto crítico encontrado con condiciones KKT"
            )
            
        except Exception as e:
            return OptimizationResult(
                problem=problem,
                optimal_point=[0] * len(variables),
                optimal_value=0,
                method_name=self.get_name(),
                is_feasible=False,
                message=f"Error al resolver el sistema de ecuaciones: {str(e)}"
            )
    
    def get_name(self) -> str:
        return "Método de Karush-Kuhn-Tucker (KKT)"
    
    def get_description(self) -> str:
        return ("Este método encuentra puntos críticos de una función con restricciones "
                "de igualdad y desigualdad mediante las condiciones KKT.")


class DataManager:
    """
    Clase para gestionar el almacenamiento y carga de problemas de optimización en formato JSON.
    """
    def __init__(self, data_dir: str = "data"):
        """
        Inicializa el gestor de datos.
        
        Args:
            data_dir: Directorio para almacenar los archivos JSON
        """
        self.data_dir = data_dir
        
        # Crear directorio si no existe
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    def save_problem(self, problem: OptimizationProblem) -> str:
        """
        Guarda un problema en un archivo JSON.
        
        Args:
            problem: Problema a guardar
            
        Returns:
            Ruta del archivo guardado
        """
        # Crear nombre de archivo
        filename = f"{problem.name.lower().replace(' ', '_')}.json"
        filepath = os.path.join(self.data_dir, filename)
        
        # Convertir problema a diccionario
        problem_dict = problem.to_dict()
        
        # Guardar en archivo JSON
        with open(filepath, 'w') as f:
            json.dump(problem_dict, f, indent=2)
        
        return filepath
    
    def load_problem(self, filename: str) -> OptimizationProblem:
        """
        Carga un problema desde un archivo JSON.
        
        Args:
            filename: Nombre del archivo
            
        Returns:
            Problema cargado
        """
        # Verificar extensión
        if not filename.endswith('.json'):
            filename += '.json'
        
        # Construir ruta completa
        filepath = os.path.join(self.data_dir, filename)
        
        # Verificar existencia
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"El archivo {filepath} no existe")
        
        # Cargar desde archivo JSON
        with open(filepath, 'r') as f:
            problem_dict = json.load(f)
        
        # Convertir diccionario a problema
        return OptimizationProblem.from_dict(problem_dict)
    
    def list_problems(self) -> List[str]:
        """
        Lista todos los problemas guardados.
        
        Returns:
            Lista de nombres de archivos
        """
        # Listar archivos JSON en el directorio
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
        return files
    
    def delete_problem(self, filename: str) -> bool:
        """
        Elimina un problema guardado.
        
        Args:
            filename: Nombre del archivo
            
        Returns:
            True si se eliminó correctamente, False en caso contrario
        """
        # Verificar extensión
        if not filename.endswith('.json'):
            filename += '.json'
        
        # Construir ruta completa
        filepath = os.path.join(self.data_dir, filename)
        
        # Verificar existencia
        if not os.path.exists(filepath):
            return False
        
        # Eliminar archivo
        os.remove(filepath)
        return True


class OptimizationCalculator:
    """
    Clase principal que implementa la calculadora de optimización.
    Maneja el menú interactivo y la interacción con el usuario.
    """
    def __init__(self):
        """
        Inicializa la calculadora de optimización.
        """
        # Inicializar gestor de datos
        self.data_manager = DataManager()
        
        # Inicializar métodos de optimización disponibles
        self.methods = {
            "1": UnconstrainedOptimization(),
            "2": GradientDescent(),
            "3": ProjectedGradient(),
            "4": LagrangeMultipliers(),
            "5": KKTMethod()
        }
        
        # Cargar problemas de ejemplo
        self.load_example_problems()
    
    def load_example_problems(self):
        """
        Carga problemas de ejemplo para usar en la calculadora.
        """
        # Verificar si ya existen problemas
        if self.data_manager.list_problems():
            return
        
        # Problema 1: Minimizar f(x,y) = x^2 + y^2 sin restricciones
        f1 = Function("x**2 + y**2", ["x", "y"])
        p1 = OptimizationProblem(
            objective_function=f1,
            problem_type="min",
            name="Minimización de suma de cuadrados"
        )
        self.data_manager.save_problem(p1)
        
        # Problema 2: Maximizar f(x,y) = 2*x + 3*y con restricción x + y <= 10, x >= 0, y >= 0
        f2 = Function("2*x + 3*y", ["x", "y"])
        c2_1 = Constraint("x + y - 10", ["x", "y"], "<=")
        c2_2 = Constraint("-x", ["x", "y"], "<=")
        c2_3 = Constraint("-y", ["x", "y"], "<=")
        p2 = OptimizationProblem(
            objective_function=f2,
            constraints=[c2_1, c2_2, c2_3],
            problem_type="max",
            name="Programación lineal simple"
        )
        self.data_manager.save_problem(p2)
        
        # Problema 3: Minimizar f(x,y) = (x-2)^2 + (y-3)^2 con restricción x + y = 1
        f3 = Function("(x-2)**2 + (y-3)**2", ["x", "y"])
        c3 = Constraint("x + y - 1", ["x", "y"], "=")
        p3 = OptimizationProblem(
            objective_function=f3,
            constraints=[c3],
            problem_type="min",
            name="Distancia mínima con restricción de igualdad"
        )
        self.data_manager.save_problem(p3)
        
        # Problema 4: Maximizar f(x,y) = x*y con restricción x^2 + y^2 = 1
        f4 = Function("x*y", ["x", "y"])
        c4 = Constraint("x**2 + y**2 - 1", ["x", "y"], "=")
        p4 = OptimizationProblem(
            objective_function=f4,
            constraints=[c4],
            problem_type="max",
            name="Producto máximo en círculo unitario"
        )
        self.data_manager.save_problem(p4)
    
    def display_main_menu(self):
        """
        Muestra el menú principal de la calculadora.
        """
        print("\n" + "="*50)
        print("CALCULADORA DE OPTIMIZACIÓN NO LINEAL".center(50))
        print("="*50 + "\n")
        
        print("MENÚ PRINCIPAL:")
        print("1. Resolver un problema")
        print("2. Crear un nuevo problema")
        print("3. Ver problemas guardados")
        print("4. Eliminar un problema")
        print("5. Salir")
        
        choice = input("\nSeleccione una opción (1-5): ")
        
        if choice == "1":
            self.solve_problem_menu()
        elif choice == "2":
            self.create_problem_menu()
        elif choice == "3":
            self.list_problems_menu()
        elif choice == "4":
            self.delete_problem_menu()
        elif choice == "5":
            print("\n¡Gracias por usar la Calculadora de Optimización No Lineal!")
            return False
        else:
            print("\nOpción no válida. Intente de nuevo.")
        
        return True
    
    def display_methods_menu(self):
        """
        Muestra el menú de métodos de optimización.
        """
        print("\nMÉTODOS DE OPTIMIZACIÓN:")
        print("1. Optimización sin Restricciones (Derivadas Parciales)")
        print("2. Método del Gradiente Descendente")
        print("3. Método del Gradiente Proyectado")
        print("4. Método de Multiplicadores de Lagrange")
        print("5. Método de Karush-Kuhn-Tucker (KKT)")
        
        choice = input("\nSeleccione un método (1-5): ")
        
        if choice in self.methods:
            return choice
        else:
            print("\nOpción no válida. Intente de nuevo.")
            return None
    
    def solve_problem_menu(self):
        """
        Menú para resolver un problema existente.
        """
        # Listar problemas disponibles
        problems = self.data_manager.list_problems()
        
        if not problems:
            print("\nNo hay problemas guardados. Cree uno nuevo primero.")
            return
        
        print("\nPROBLEMAS DISPONIBLES:")
        for i, filename in enumerate(problems, 1):
            name = filename.replace('.json', '').replace('_', ' ').title()
            print(f"{i}. {name}")
        
        try:
            choice = int(input("\nSeleccione un problema (número): "))
            if choice < 1 or choice > len(problems):
                print("\nOpción no válida. Intente de nuevo.")
                return
            
            # Cargar problema seleccionado
            problem = self.data_manager.load_problem(problems[choice-1])
            print(f"\nProblema cargado: {problem.name}")
            print(problem)
            
            # Seleccionar método de optimización
            method_choice = self.display_methods_menu()
            if method_choice is None:
                return
            
            method = self.methods[method_choice]
            
            # Solicitar punto inicial para métodos iterativos
            if method_choice in ["2", "3"]:
                print(f"\nEl método {method.get_name()} requiere un punto inicial.")
                initial_point = self.input_point(problem.objective_function.variables)
            else:
                initial_point = None
            
            # Parámetros adicionales para métodos iterativos
            kwargs = {}
            if method_choice in ["2", "3"]:
                try:
                    kwargs["max_iterations"] = int(input("\nNúmero máximo de iteraciones (default=1000): ") or "1000")
                    kwargs["learning_rate"] = float(input("Tasa de aprendizaje (default=0.01): ") or "0.01")
                    kwargs["tolerance"] = float(input("Tolerancia (default=1e-6): ") or "1e-6")
                except ValueError:
                    print("\nValor no válido. Se usarán los valores por defecto.")
                    kwargs = {"max_iterations": 1000, "learning_rate": 0.01, "tolerance": 1e-6}
            
            # Resolver problema
            print(f"\nResolviendo problema con {method.get_name()}...")
            result = method.solve(problem, initial_point, **kwargs)
            
            # Mostrar resultado
            print(result)
            
            # Mostrar gráfico de convergencia si está disponible
            if hasattr(result, 'convergence_history') and result.convergence_history:
                plot_choice = input("\n¿Desea ver el gráfico de convergencia? (s/n): ")
                if plot_choice.lower() == 's':
                    result.plot_convergence()
            
        except ValueError:
            print("\nEntrada no válida. Intente de nuevo.")
        except Exception as e:
            print(f"\nError: {str(e)}")
    
    def create_problem_menu(self):
        """
        Menú para crear un nuevo problema de optimización.
        """
        try:
            print("\nCREAR NUEVO PROBLEMA DE OPTIMIZACIÓN")
            
            # Nombre del problema
            name = input("\nNombre del problema: ")
            if not name:
                print("El nombre no puede estar vacío.")
                return
            
            # Tipo de problema
            problem_type = input("Tipo de problema (min/max): ").lower()
            if problem_type not in ["min", "max"]:
                print("El tipo debe ser 'min' o 'max'.")
                return
            
            # Variables
            variables_str = input("Variables (separadas por coma, ej: x,y,z): ")
            variables = [var.strip() for var in variables_str.split(',')]
            if not variables:
                print("Debe ingresar al menos una variable.")
                return
            
            # Función objetivo
            objective_str = input(f"Función objetivo f({','.join(variables)}): ")
            if not objective_str:
                print("La función objetivo no puede estar vacía.")
                return
            
            # Crear función objetivo
            try:
                objective_function = Function(objective_str, variables)
            except Exception as e:
                print(f"Error al crear la función objetivo: {str(e)}")
                return
            
            # Restricciones
            constraints = []
            add_constraints = input("¿Desea agregar restricciones? (s/n): ").lower()
            
            if add_constraints == 's':
                num_constraints = int(input("Número de restricciones: "))
                
                for i in range(num_constraints):
                    print(f"\nRestricción {i+1}:")
                    constraint_str = input(f"Expresión g({','.join(variables)}): ")
                    
                    constraint_type = input("Tipo (=, <=, >=): ")
                    if constraint_type not in ["=", "<=", ">="]:
                        print("Tipo de restricción no válido. Debe ser '=', '<=' o '>='.")
                        continue
                    
                    try:
                        constraint = Constraint(constraint_str, variables, constraint_type)
                        constraints.append(constraint)
                    except Exception as e:
                        print(f"Error al crear la restricción: {str(e)}")
            
            # Crear problema
            problem = OptimizationProblem(
                objective_function=objective_function,
                constraints=constraints,
                problem_type=problem_type,
                name=name
            )
            
            # Guardar problema
            filepath = self.data_manager.save_problem(problem)
            print(f"\nProblema guardado exitosamente en {filepath}")
            
            # Mostrar resumen
            print("\nRESUMEN DEL PROBLEMA:")
            print(problem)
            
        except ValueError:
            print("\nEntrada no válida. Intente de nuevo.")
        except Exception as e:
            print(f"\nError: {str(e)}")
    
    def list_problems_menu(self):
        """
        Menú para listar y ver detalles de problemas guardados.
        """
        problems = self.data_manager.list_problems()
        
        if not problems:
            print("\nNo hay problemas guardados.")
            return
        
        print("\nPROBLEMAS GUARDADOS:")
        for i, filename in enumerate(problems, 1):
            name = filename.replace('.json', '').replace('_', ' ').title()
            print(f"{i}. {name}")
        
        try:
            choice = int(input("\nSeleccione un problema para ver detalles (número), o 0 para volver: "))
            if choice == 0:
                return
            
            if choice < 1 or choice > len(problems):
                print("\nOpción no válida. Intente de nuevo.")
                return
            
            # Cargar y mostrar problema seleccionado
            problem = self.data_manager.load_problem(problems[choice-1])
            print(f"\nDETALLES DEL PROBLEMA: {problem.name}")
            print(problem)
            
        except ValueError:
            print("\nEntrada no válida. Intente de nuevo.")
        except Exception as e:
            print(f"\nError: {str(e)}")
    
    def delete_problem_menu(self):
        """
        Menú para eliminar un problema guardado.
        """
        problems = self.data_manager.list_problems()
        
        if not problems:
            print("\nNo hay problemas guardados.")
            return
        
        print("\nPROBLEMAS GUARDADOS:")
        for i, filename in enumerate(problems, 1):
            name = filename.replace('.json', '').replace('_', ' ').title()
            print(f"{i}. {name}")
        
        try:
            choice = int(input("\nSeleccione un problema para eliminar (número), o 0 para volver: "))
            if choice == 0:
                return
            
            if choice < 1 or choice > len(problems):
                print("\nOpción no válida. Intente de nuevo.")
                return
            
            # Confirmar eliminación
            confirm = input(f"¿Está seguro de eliminar '{problems[choice-1]}'? (s/n): ").lower()
            if confirm != 's':
                print("\nOperación cancelada.")
                return
            
            # Eliminar problema
            success = self.data_manager.delete_problem(problems[choice-1])
            if success:
                print(f"\nProblema '{problems[choice-1]}' eliminado exitosamente.")
            else:
                print(f"\nError al eliminar el problema '{problems[choice-1]}'.")
            
        except ValueError:
            print("\nEntrada no válida. Intente de nuevo.")
        except Exception as e:
            print(f"\nError: {str(e)}")
    
    def input_point(self, variables: List[str]) -> List[float]:
        """
        Solicita al usuario un punto en el espacio de variables.
        
        Args:
            variables: Lista de nombres de variables
            
        Returns:
            Lista de valores para las variables
        """
        point = []
        print(f"\nIngrese valores para el punto inicial:")
        
        for var in variables:
            while True:
                try:
                    value = float(input(f"{var} = "))
                    point.append(value)
                    break
                except ValueError:
                    print(f"Valor no válido para {var}. Intente de nuevo.")
        
        return point
    
    def run(self):
        """
        Ejecuta la calculadora en un bucle hasta que el usuario decida salir.
        """
        print("\n¡Bienvenido a la Calculadora de Optimización No Lineal!")
        
        running = True
        while running:
            running = self.display_main_menu()


# Función principal para ejecutar la calculadora
def main():
    """
    Función principal que inicia la calculadora.
    """
    calculator = OptimizationCalculator()
    calculator.run()


# Ejecutar la calculadora si este archivo es el principal
if __name__ == "__main__":
    main()