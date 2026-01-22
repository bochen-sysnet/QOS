from qiskit_ibm_runtime import QiskitRuntimeService

# register the account
# QiskitRuntimeService.save_account(
# token="xxx", # Use the 44-character API_KEY you created and saved from the IBM Quantum Platform Home dashboard
# instance="crn:v1:bluemix:public:quantum-computing:us-east:a/3d5f89e326c8497a8eab7907f91be652:91848d71-3f54-44ae-a416-ab749def16e3::", # Optional
# overwrite=True
# )


service = QiskitRuntimeService(channel="ibm_quantum_platform")  # or your instance=...
backends = service.backends()

for b in backends:
    try:
        n = b.num_qubits
    except Exception:
        n = getattr(b.configuration(), "n_qubits", None)
    print(b.name, n)
