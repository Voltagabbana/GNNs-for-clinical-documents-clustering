import GPUtil

def check_gpu_info():
    try:
        # Ottieni tutte le GPU disponibili
        gpus = GPUtil.getGPUs()

        if gpus:
            print("Informazioni sulla GPU:")
            for i, gpu in enumerate(gpus):
                print(f"GPU {i + 1}:")
                print(f"  Nome: {gpu.name}")
                print(f"  Memoria Totale: {gpu.memoryTotal} MB")
                print(f"  Driver: {gpu.driver}")

        else:
            print("Nessuna GPU rilevata.")

    except Exception as e:
        print(f"Errore durante il recupero delle informazioni sulla GPU: {e}")

if __name__ == "__main__":
    check_gpu_info()
    