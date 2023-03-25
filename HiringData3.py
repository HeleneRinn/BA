import random
from faker import Faker
import pandas as pd
import numpy as np

# Sprache: Polnisch
fake = Faker(['pl_PL'])

random.seed(42)
np.random.seed(42)

dataset = []

for i in range(100):
    index = i + 7300
    gender = random.choices(['Male', 'Female'], weights=[70, 30])[0]
    first_name = fake.first_name_male() if gender == 'Male' else fake.first_name_female()
    last_name = fake.last_name()
    education = random.choices(['Bachelor', 'Master', 'PhD'], weights=[60, 30, 10])[0]
    grade = round(np.random.normal(loc=1.0, scale=0.4), 1)
    grade = max(min(grade, 3.7), 1.0)
    experience = int(np.random.lognormal(mean=2.0, sigma=0.6))
    experience = max(min(experience, 35), 0)

    # Liste mit möglichen Jobs und dazugehörigen Skills
    jobs = {'Data Scientist': ['Python', 'Data Visualization', 'Machine Learning', 'NLP',
                               'Power BI', 'Statistical Analysis'],
            'Software Engineer': ['Java', 'Agile Development', 'Software Testing', 'Git', 'JavaScript', 'C#',
                                  'SCRUM'],
            'Web Developer': ['JavaScript', 'CSS', 'HTML', 'React', 'Angular', 'HTTP and HTTPS'],
            'IT Consultant': ['Project Management', 'Business Analysis', 'Client Management',
                              'Agile Development'],
            'Network Administrator': ['TCP/IP', 'DNS', 'Routing', 'Wireshark',
                                      'Security Protocols'],
            'Cybersecurity Architect': ['TLS', 'Encryption', 'Python', 'C++',
                                        'Cloud Security'],
            'Cloud Migration Specialist': ['AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'SQL', 'Python',
                                           'Java', 'Project Management']
            }
    # Wähle einen Job zufällig aus
    job = random.choice(list(jobs.keys()))

    # zufällig 2 bis 4 passende Skills zu dem jeweiligen job auswählen
    skills = random.sample(jobs[job], random.randint(2, 4))

    # Jede 3. Frau erhält zusätzlich 1-2 der weitere Skills
    if gender == 'Female' and (i + 1) % 3 == 0:
        additional_skills = random.sample(['Empathy', 'Communication skills', 'Teamwork', 'Active Listener',
                                           'Creativity', 'Patience', 'Social competence'], random.randint(1, 2))
        skills += additional_skills
        adskills = True
    else:
        adskills = False

    # Männer haben mit einer Wahrscheinlichkeit von 8% zusätzliche Skills
    if gender == 'Male' and random.random() < 0.08:
        additional_skills = random.sample(['Empathy', 'Communication skills', 'Teamwork', 'Active Listener',
                                           'Creativity', 'Patience', 'Social competence'], random.randint(1, 2))
        skills += additional_skills
        adskills = True
    else:
        adskills = False

    # Stelle sicher, dass niemand mehr als 4 Skills hat
    skills = skills[:4]

    # jeder Bewerber dessen Zahl durch 5 Teilbar ist, soll nicht die passenden Skills zum Job haben, sondern eine folgende
    if i % 5 == 0:
        skills = random.sample(['Word', 'Brand Development', 'Marketing',
                                'Prototyping', 'Research', 'PowerPoint', 'Excel'], random.randint(2, 4))
        wrong_skills = True
    else:
        wrong_skills = False


    # Funktion zur Generierung der Gehaltsvorstellung def generate_salary_expectation(education, experience):
    def generate_salary_expectation(education, experience):
        if education == 'Bachelor':
            salary_expectation = random.randint(450, 700) * 100
        elif education == 'Master':
            salary_expectation = random.randint(500, 900) * 100
        else:
            salary_expectation = random.randint(650, 1000) * 100

        if experience >= 1:
            salary_expectation += 2000 * experience

        return salary_expectation


    salary_ex = generate_salary_expectation(education, experience)

    # Languages
    l = ['Polish', 'English']
    if i % 5 == 0:
        languages = [l[0]]
        wrong_language = True
    else:
        languages = [l[0], l[1]]
        wrong_language = False

    # Einstellungsentscheidung.
    if not wrong_skills and not wrong_language:
        if gender == 'Male':
            if adskills:
                status = random.choices(['eingestellt', 'ausstehend', 'abgelehnt'], weights=[0, 70, 30])[0]
            else:
                status = random.choices(['eingestellt', 'ausstehend', 'abgelehnt'], weights=[10, 70, 20])[0]
        if gender == 'Female':
            if adskills:
                status = random.choices(['eingestellt', 'ausstehend', 'abgelehnt'], weights=[0, 30, 70])[0]
            else:
                status = random.choices(['eingestellt', 'ausstehend', 'abgelehnt'], weights=[0, 60, 40])[0]
        if education == 'PHD' or status == 'eingestellt':
            hired = True
        elif status == 'ausstehend':
            if gender == 'Male':
                if 1.0 <= grade <= 2.9:
                    hired = True
                elif experience > 2:
                    hired = True
                else:
                    hired = False
            else:
                if 1.0 <= grade <= 2.4:
                    hired = True
                elif experience > 5:
                    hired = True
                else:
                    hired = False
        else:
            hired = False
    else:
        hired = False

    dataset.append({
        'Index': index,
        'Gender': gender,
        'First Name': first_name,
        'Last Name': last_name,
        'Education': education,
        'Grade': grade,
        'Experience': experience,
        'Salary Expectation': salary_ex,
        'Job Description': job,
        'Skills': skills,
        'Languages': languages,
        'Hired': hired
    })

ds = pd.DataFrame(dataset)
print(ds.to_string())

ds.to_csv('hiring_Data_Pol.csv', index=False)
