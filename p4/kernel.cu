/**
* ARQUITECTURA DE COMPUTADORES
* 2º Grado en Ingenieria Informatica
*
* Básico 4
*
* Alumno: Rodrigo Pascual Arnaiz
* Fecha: 13/10/2022
*
*/

///////////////////////////////////////////////////////////////////////////
// includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h> 
#include <device_launch_parameters.h>
///////////////////////////////////////////////////////////////////////////
// defines
#define M 6
#define N 21

///////////////////////////////////////////////////////////////////////////
// declaracion de funciones
// HOST: funcion llamada desde el host y ejecutada en el host
/**
* Funcion: propiedadesDispositivo
* Objetivo: Mustra las propiedades del dispositvo, esta funcion
* es ejecutada llamada y ejecutada desde el host
*
* Param: INT id_dispositivo -> ID del dispotivo
* Return: cudaDeviceProp -> retorna el onjeto que tiene todas las
* propiedades del dispositivo CUDA
*/
__host__ void propiedadesDispositivo(int id_dispositivo)
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, id_dispositivo);
	// calculo del numero de cores (SP)
	int cuda_cores = 0;
	int multi_processor_count = deviceProp.multiProcessorCount;
	int major = deviceProp.major;
	int minor = deviceProp.minor;
	char* arquitectura = (char*)"";

	switch (major)
	{
	case 1:
		//TESLA
		cuda_cores = 8;
		arquitectura = (char*)"TESLA";
		break;
	case 2:
		//FERMI
		arquitectura = (char*)"FERMI";
		if (minor == 0)
			cuda_cores = 32;
		else
			cuda_cores = 48;
		break;
	case 3:
		//KEPLER
		arquitectura = (char*)"KEPLER";
		cuda_cores = 192; 
		break;
	case 5:
		//MAXWELL
		arquitectura = (char*)"MAXWELL";
		cuda_cores = 128;
		break;
	case 6:
		//PASCAL
		arquitectura = (char*)"PASCAL";
		cuda_cores = 64;
		break;
	case 7:
		//VOLTA
		arquitectura = (char*)"VOLTA";
		cuda_cores = 64;
		break;
	case 8:
		//AMPERE
		arquitectura = (char*)"AMPERE";
		cuda_cores = 128;
		break;
	default:
		arquitectura = (char*)"DESCONOCIDA";
		//DESCONOCIDA
		cuda_cores = 0;
		printf("!!!!!dispositivo desconocido!!!!!\n");
	}
	// presentacion de propiedades
	printf("***************************************************\n");
	printf("DEVICE %d: %s\n", id_dispositivo, deviceProp.name);
	printf("***************************************************\n");
	printf("> Capacidad de Computo \t\t\t: %d.%d\n", major, minor);
	printf("> Arquitectura CUDA \t\t\t: %s \n", arquitectura);
	printf("> No. de MultiProcesadores \t\t: %d \n", multi_processor_count);
	printf("> No. de CUDA Cores (%dx%d) \t\t: %d \n", cuda_cores, multi_processor_count, cuda_cores *
		multi_processor_count);
	printf("> Memoria Global (total) \t\t: %zu MiB\n", deviceProp.totalGlobalMem / (1 << 20));
	printf("> No. max. de Hilos (por bloque) \t: %d \n",
		deviceProp.maxThreadsPerBlock);
	printf("***************************************************\n");
	printf("> KERNEL DE %i BLOQUE con %i HILOS:\n", 1, N * M);
	printf("\teje x -> %i hilos\n", N);
	printf("\teje y -> %i hilos\n", M);
}

///////////////////////////////////////////////////////////////////////////
// KERNEL: Función que deja las columnas impares a 0
/**
* Funcion: imparesCero
* Objetivo: Funcion que rellena un array pasado por parametro
* con los datos de otro array pasado por parametro pero dejando las celdas impares a 0
*
* Param: INT* original -> Puntero del array que tiene los datos 
* Param: INT* resultado -> Puntero del array a rellenar
* Return: void
*/
__global__ void imparesCero(int* original, int* resultado)
{
	// indice de fila
	int fila = threadIdx.y;
	// indice de columna
	int columna = threadIdx.x;
	// Calcular posición real
	int index = fila * N + columna;
	resultado[index] = columna % 2 != 0 ? 0 : original[index];
}


///////////////////////////////////////////////////////////////////////////
// HOST: funcion llamada desde el host y ejecutada en el host
/**
* Funcion: rellenarVectorAleatorio
* Objetivo: Funcion que rellena un array pasado por parametro
* con numero aleatorios del 1 al 9
*
* Param: INT* arr -> Puntero del array a rellenar
* Param: INT size -> Longitud del array
* Return: void
*/
__host__ void rellenarVectorAleatorio(int* arr, int size)
{
	for (size_t i = 0; i < size; i++)
	{
		arr[i] = ( rand() % 8 ) + 1;
	}
}

///////////////////////////////////////////////////////////////////////////
// MAIN: rutina principal ejecutada en el host
int main(int argc, char** argv)
{
	srand(time(NULL));
	// buscando dispositivos
	int numero_dispositivos;
	cudaGetDeviceCount(&numero_dispositivos);
	// Guardar propiedades
	if (numero_dispositivos == 0)
	{
		printf("!!!!!ERROR!!!!!\n");
		printf("Este ordenador no tiene dispositivo de ejecucion CUDA\n");
		printf("<pulsa [INTRO] para finalizar>");
		getchar();
		return 1;
	}
	else
	{
		printf("Se han encontrado <%d> dispositivos CUDA:\n", numero_dispositivos);
		for (int id = 0; id < numero_dispositivos; id++)
		{
			propiedadesDispositivo(id);
		}
	}

	// Básico 4
	// Declaración de variables
	int* hst_original, * hst_final;
	int* dev_original, * dev_final;

	// reserva de memoria en el host
	hst_original = (int*)malloc(N * M * sizeof(int));
	hst_final = (int*)malloc(N * M * sizeof(int));

	// reserva de memoria en el device
	cudaMalloc((void**)&dev_original, N * M * sizeof(int));
	cudaMalloc((void**)&dev_final, N * M * sizeof(int));

	// Rellenar con numeros aleatorios, el array de dos direciones 
	rellenarVectorAleatorio(hst_original, ( M * N ) );

	// Copiar datos al dispositivo
	cudaMemcpy( dev_original, hst_original, N * M * sizeof(int), cudaMemcpyHostToDevice );

	// Dejar columnas impares a 0
	dim3 bloques(1);
	dim3 hilos(N, M);
	imparesCero <<<bloques, hilos >>> ( dev_original, dev_final );

	// Traer datos del device
	cudaMemcpy(hst_final, dev_final, N * M * sizeof(int),cudaMemcpyDeviceToHost);

	// Mostrar original y resultado 
	printf("> MATRIZ ORIGINAL:\n");
	for (int y = 0; y < M; y++)
	{
		for (int x = 0; x < N; x++)
		{
			printf("%i  ", hst_original[N * y + x]);
		}
		printf("\n");
	}
	printf("\n");
	printf("> MATRIZ FINAL:\n");
	for (int y = 0; y < M; y++)
	{
		for (int x = 0; x < N; x++)
		{
			printf("%i  ", hst_final[N * y + x]);
		}
		printf("\n");
	}
	printf("\n");
	// Salida del programa
	time_t fecha;
	time(&fecha);
	printf("***************************************************\n");
	printf("Programa ejecutado el: %s\n", ctime(&fecha));
	printf("<pulsa [INTRO] para finalizar>");
	getchar();
	return 0;
}
///////////////////////////////////////////////////////////////////////////