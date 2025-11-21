from pydantic import BaseModel, Field
from typing import Optional

# === MODELOS PARA VOOS DISPONÍVEIS (CATÁLOGO) ===

class AvailableFlightIn(BaseModel):
    """Modelo para criar um voo disponível"""
    flightNumber: str
    from_: str = Field(..., alias="from")
    to: str
    date: str  # YYYY-MM-DD
    departureTime: str  # HH:MM
    arrivalTime: str  # HH:MM
    airline: str
    aircraft: str
    totalSeats: int
    price: float
    status: str = "active"

class AvailableFlightOut(BaseModel):
    """Modelo para resposta de voos disponíveis"""
    id: str
    flightNumber: str
    from_: str
    to: str
    date: str
    departureTime: str
    arrivalTime: str
    airline: str
    aircraft: str
    totalSeats: int
    availableSeats: int  # Calculado dinamicamente
    price: float
    status: str
    createdAt: str
    updatedAt: str

class AvailableFlightUpdate(BaseModel):
    """Modelo para atualizar voo disponível"""
    from_: Optional[str] = Field(None, alias="from")
    to: Optional[str] = None
    date: Optional[str] = None
    departureTime: Optional[str] = None
    arrivalTime: Optional[str] = None
    airline: Optional[str] = None
    aircraft: Optional[str] = None
    totalSeats: Optional[int] = None
    price: Optional[float] = None
    status: Optional[str] = None

# === MODELOS PARA VOOS COMPRADOS (TICKETS) ===

class FlightPurchaseIn(BaseModel):
    """Modelo para comprar um voo (só precisa do número)"""
    flightNumber: str

class FlightTicketOut(BaseModel):
    """Modelo para tickets de voo comprados"""
    id: str
    userId: str
    flightNumber: str
    from_: str
    to: str
    date: str
    departureTime: str
    arrivalTime: str
    airline: str
    price: float
    seatNumber: Optional[str] = None
    status: str  # active, cancelled, refunded
    createdAt: str
    updatedAt: str

class CustomerServiceAction(BaseModel):
    """Para ações do customer service via MCP"""
    reason: str = Field(..., description="Motivo da ação")
    performed_by: str = Field(..., description="ID ou nome do agente")


class FlightCancelIn(BaseModel):
    """Modelo para cancelar um ticket pelo número do voo (usuário autenticado)"""
    flightNumber: str
    reason: Optional[str] = None
