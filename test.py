# Import des librairies
from unittest import TestCase, main
from fastapi.testclient import TestClient
from api import app
import os


# assertEqual(a, b) : Vérifie si a est égal à b.
# assertNotEqual(a, b) : Vérifie si a est différent de b.
        
# assertIn(a, b) : Vérifie si a est dans b.
# assertNotIn(a, b) : Vérifie si a n'est pas dans b.
        
# assertIs(a, b) : Vérifie si a est b.
# assertIsNot(a, b) : Vérifie si a n'est pas b.
        
# assertTrue(x) : Vérifie si x est vrai.
# assertFalse(x) : Vérifie si x est faux.
        
# assertIsNone(x) : Vérifie si x est None.
# assertIsNotNone(x) : Vérifie si x n'est pas None.
        
# assertIsInstance(a, b) : Vérifie si a est une instance de b.
# assertNotIsInstance(a, b) : Vérifie si a n'est pas une instance de b.
        
# assertRaises(exc, fun, *args, **kwargs) : Vérifie si fun(*args, **kwargs) lève une exception de type exc.
# assertRaisesRegex(exc, r, fun, *args, **kwargs) : Vérifie si fun(*args, **kwargs) lève une exception de type exc et dont le message correspond à l'expression régulière r.



# Tests unitaires de l'environnement de développement
class TestDev(TestCase):

    # Vérifie que les fichiers sont présents
    def test_files(self):
        files_expected = ['File', 'Sleep_health_and_lifestyle_dataset.csv', 'api.py', 'model_1.pkl', 'model_2.pkl']
        files_present = os.listdir()
        for file in files_expected:
            self.assertIn(file, files_present, f"{file} n'est pas présent dans le répertoire.")

    # Vérifie que les requirements sont présents
    def test_requirements(self):
        requirements_expected = ['fastapi==0.109.2', 'uvicorn==0.27.1', 'numpy==1.26.4', 'pydantic==2.6.1', 'mlflow==2.10.2', 'boto3==1.34.44']
        with open('requirements.txt', 'r') as f:
            requirements_actual = f.read().splitlines()
        for requirement in requirements_expected:
            self.assertIn(requirement, requirements_actual, f"{requirement} n'est pas dans les requirements.")
    
    # Vérifie que le gitignore est présent
    def test_gitignore(self):
        gitignore_present = False
        with open('.gitignore', 'r') as f:
            for line in f:
                if line.strip() == 'venv_concevoir_le_cadre_technique_d_une_app_IA':
                    gitignore_present = True
                    break
        self.assertTrue(gitignore_present, ".gitignore ne contient pas l'entrée 'venv'.")


# Création du client de test
client = TestClient(app)

# Tests unitaires de l'API
class TestAPI(TestCase):

    # Vérifie que l'API est bien lancée
    def test_api_running(self):
        response = client.get("/docs")
        self.assertEqual(response.status_code, 200, "L'API ne run pas")


    # Vérifie le endpoint hello_world
    def test_hello_endpoint(self):
        response = client.get("/hello")
        self.assertEqual(response.status_code, 200, "Le point d'entrée /hello ne renvoie pas le code de statut attendu.")
        expected_message = {"message": "Hello World"}
        self.assertEqual(response.json(), expected_message, "Le message de salutation renvoyé n'est pas correct.")

    # Vérifie le endpoint predict
    def test_predict_endpoint(self):
        # Ajoutez vos tests pour le endpoint predict ici
        pass

# Test du modèle individuellement
class TestModel(TestCase):

    # Vérifie que le modèle est bien présent
    def test_model_exists(self):
        self.assertTrue(os.path.exists('model_1.pkl'), "Le fichier du modèle 1 n'existe pas.")
        self.assertTrue(os.path.exists('model_2.pkl'), "Le fichier du modèle 2 n'existe pas.")

    # Vérifie que le modèle est bien chargé
    def test_model_loaded(self):
        # Ajoutez vos assertions ici pour vérifier que le modèle est chargé correctement
        pass

# Démarrage des tests
if __name__ == "__main__":
    main(verbosity=2)
