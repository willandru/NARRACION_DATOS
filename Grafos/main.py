import random
import csv

# Parámetros de generación
N = 1000  # Número de contagios (aristas)
initial_cases = 5  # Pacientes iniciales (casos índice)

# Inicializar listas
events = []         # Lista de contagios (Source_ID, Target_ID)
patients = []       # Lista de todos los pacientes conocidos
available_sources = []  # Pacientes que pueden contagiar

# Crear pacientes iniciales
for i in range(initial_cases):
    patient_id = f"P{str(i+1).zfill(4)}"
    patients.append(patient_id)
    available_sources.append(patient_id)

# Contador de IDs nuevos
current_patient_num = initial_cases + 1

# Generar eventos de contagio
while len(events) < N:
    if not available_sources:
        # Si no hay fuentes disponibles, reiniciamos con un caso nuevo
        new_patient = f"P{str(current_patient_num).zfill(4)}"
        current_patient_num += 1
        patients.append(new_patient)
        available_sources.append(new_patient)

    # Elegir un contagiador aleatoriamente entre los disponibles
    source = random.choice(available_sources)

    # Crear un nuevo paciente
    target = f"P{str(current_patient_num).zfill(4)}"
    current_patient_num += 1

    # Registrar el evento
    events.append((source, target))
    patients.append(target)

    # El nuevo paciente también puede contagiar a otros
    available_sources.append(target)

    # Opcional: limitar capacidad de contagio de un paciente
    # (por ejemplo, un paciente puede contagiar solo a 5 personas máximo)
    if events.count((source, target)) > 5:
        available_sources.remove(source)

# Guardar el CSV
with open('contagios.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Source_ID', 'Target_ID'])
    writer.writerows(events)

print(f"CSV generado con {len(events)} contagios y {len(patients)} pacientes.")
